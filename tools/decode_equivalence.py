#!/usr/bin/env python3
"""Measure how closely each candidate NEW decode reproduces the OLD detections.

Runs the PyTorch head once per frame, then applies the reference decode and
three variants of the CUDA-kernel decode to the SAME tensors. Reports, for each
variant, what fraction of OLD detections it reproduces and how many extra boxes
it emits. This isolates the effect of each individual difference.

    python3 tools/decode_equivalence.py --bag <bag_dir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'reference' / 'lidar_object_detection_py'))

import torch  # noqa: E402
from rosbags.highlevel import AnyReader  # noqa: E402
from lidar_object_detection.model.pointpillars import PointPillars  # noqa: E402
from lidar_object_detection.model.anchors import anchors2bboxes  # noqa: E402
from lidar_object_detection.ops import nms_cuda  # noqa: E402

sys.path.insert(0, str(REPO / 'tools'))
from decode_stage_audit import cloud_to_xyzi, point_range_filter, greedy_nms  # noqa: E402

NCLASSES = 3
MATCH_M = 0.5


def match(old_xy, old_cls, new_xy, new_cls):
    """Fraction of OLD boxes with a same-class NEW box within MATCH_M."""
    if len(old_xy) == 0:
        return 0, 0
    if len(new_xy) == 0:
        return 0, len(old_xy)
    d = np.linalg.norm(old_xy[:, None, :] - new_xy[None, :, :], axis=-1)
    d[old_cls[:, None] != new_cls[None, :]] = np.inf
    return int((d.min(1) <= MATCH_M).sum()), len(old_xy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--weights',
                    default=str(REPO / 'reference/lidar_object_detection_py/weights/epoch_160.pth'))
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--score-thr', type=float, default=0.1)
    ap.add_argument('--nms-thr', type=float, default=0.01)
    a = ap.parse_args()

    dev = torch.device('cuda')
    model = PointPillars(nclasses=NCLASSES).to(dev)
    model.load_state_dict(torch.load(a.weights))
    model.eval()
    NMS_PRE, MAX_NUM = model.nms_pre, model.max_num

    variants = ['A: no caps (current)', 'B: +topk/max_num', 'C: B +per-class thr']
    hit = {v: 0 for v in variants}
    tot = {v: 0 for v in variants}
    cnt = {v: [] for v in variants}
    old_n = []

    with AnyReader([Path(a.bag)]) as reader:
        conns = [c for c in reader.connections if 'PointCloud2' in c.msgtype]
        n = 0
        for conn, _, raw in reader.messages(connections=conns):
            if n >= a.frames:
                break
            pts = point_range_filter(cloud_to_xyzi(reader.deserialize(raw, conn.msgtype)))
            with torch.no_grad():
                pil, coors, npts = model.pillar_layer([torch.from_numpy(pts).to(dev)])
                x = model.neck(model.backbone(model.pillar_encoder(pil, coors, npts)))
                cls_p, box_p, dir_p = model.head(x)
            fms = torch.tensor(list(cls_p.size()[-2:]), device=dev)
            an = model.anchors_generator.get_multi_anchors(fms).reshape(-1, 7)
            c = cls_p[0].permute(1, 2, 0).reshape(-1, NCLASSES).sigmoid()
            b = box_p[0].permute(1, 2, 0).reshape(-1, 7)

            # ---- OLD (reference, verbatim)
            inds = c.max(1)[0].topk(NMS_PRE)[1]
            c_o, bb = c[inds], anchors2bboxes(an[inds], b[inds])
            xy, lw = bb[:, [0, 1]], bb[:, [3, 4]]
            bb2d = torch.cat([xy - lw / 2, xy + lw / 2, bb[:, 6:]], -1)
            oxy, ocls, oscore = [], [], []
            for i in range(NCLASSES):
                s = c_o[:, i]
                m = s > a.score_thr
                if m.sum() == 0:
                    continue
                k = nms_cuda(boxes=bb2d[m].contiguous(), scores=s[m].contiguous(),
                             thresh=a.nms_thr, pre_maxsize=None, post_max_size=None)
                sel = bb[m][k]
                oxy.append(sel[:, :2].cpu().numpy())
                ocls.append(np.full(len(k), i))
                oscore.append(s[m][k].cpu().numpy())
            oxy = np.concatenate(oxy) if oxy else np.zeros((0, 2))
            ocls = np.concatenate(ocls) if len(ocls) else np.zeros(0, int)
            oscore = np.concatenate(oscore) if len(oscore) else np.zeros(0)
            if len(oxy) > MAX_NUM:
                keep = np.argsort(-oscore)[:MAX_NUM]
                oxy, ocls = oxy[keep], ocls[keep]
            old_n.append(len(oxy))

            def run_new(topk, per_class):
                if per_class:
                    rank = c.max(1)[0]
                    idx = rank.topk(topk)[1] if topk else torch.arange(len(c), device=dev)
                    cc, bbn = c[idx], anchors2bboxes(an[idx], b[idx])
                    pairs = []
                    for i in range(NCLASSES):
                        s = cc[:, i]
                        m = s > a.score_thr
                        if m.sum() == 0:
                            continue
                        pairs.append((i, bbn[m].cpu().numpy(), s[m].cpu().numpy()))
                else:
                    sc, cid = c.max(1)
                    m = sc >= a.score_thr
                    idx = torch.nonzero(m).squeeze(1)
                    if topk:
                        order = sc[idx].topk(min(topk, len(idx)))[1]
                        idx = idx[order]
                    bbn = anchors2bboxes(an[idx], b[idx]).cpu().numpy()
                    sn, cn = sc[idx].cpu().numpy(), cid[idx].cpu().numpy()
                    pairs = [(i, bbn[cn == i], sn[cn == i]) for i in range(NCLASSES)
                             if (cn == i).sum()]
                nxy, ncls, nsc = [], [], []
                for i, bx, s in pairs:
                    q = np.concatenate([bx[:, :2] - bx[:, 3:5] / 2,
                                        bx[:, :2] + bx[:, 3:5] / 2], -1)
                    k = greedy_nms(q, s, a.nms_thr)
                    nxy.append(bx[k, :2]); ncls.append(np.full(len(k), i)); nsc.append(s[k])
                if not nxy:
                    return np.zeros((0, 2)), np.zeros(0, int)
                nxy = np.concatenate(nxy); ncls = np.concatenate(ncls); nsc = np.concatenate(nsc)
                if topk and len(nxy) > MAX_NUM:
                    k = np.argsort(-nsc)[:MAX_NUM]
                    nxy, ncls = nxy[k], ncls[k]
                return nxy, ncls

            for v, (tk, pc) in zip(variants, [(None, False), (NMS_PRE, False), (NMS_PRE, True)]):
                nxy, ncls = run_new(tk, pc)
                h, t = match(oxy, ocls, nxy, ncls)
                hit[v] += h; tot[v] += t; cnt[v].append(len(nxy))
            n += 1
            print(f"  frame {n:2d}  OLD={old_n[-1]:3d}  " +
                  "  ".join(f"{v.split(':')[0]}={cnt[v][-1]:3d}" for v in variants), flush=True)

    print(f"\nOLD detections/frame: {np.mean(old_n):.1f}\n")
    print(f"{'variant':24s} {'det/frame':>10s} {'OLD reproduced':>16s}")
    for v in variants:
        print(f"{v:24s} {np.mean(cnt[v]):10.1f} {100*hit[v]/max(tot[v],1):15.1f}%")


if __name__ == '__main__':
    main()
