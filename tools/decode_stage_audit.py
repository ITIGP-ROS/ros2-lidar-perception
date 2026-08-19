#!/usr/bin/env python3
"""Isolate WHERE the old and new pipelines' detection counts diverge.

Both pipelines run the same network (cls_preds correlation 0.999912). The
detections differ (83.1 vs 43.0 objects/frame). This script runs the PyTorch
head ONCE per frame and then applies BOTH decode paths to the SAME raw tensors,
counting survivors at every stage. Whatever is shared cannot be the cause; the
stage where the counts split is the cause.

    python3 tools/decode_stage_audit.py --bag <bag_dir> --weights <epoch_160.pth>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REF = REPO / 'reference' / 'lidar_object_detection_py'
for p in (str(REF),):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from rosbags.highlevel import AnyReader  # noqa: E402
from lidar_object_detection.model.pointpillars import PointPillars  # noqa: E402
from lidar_object_detection.model.anchors import anchors2bboxes  # noqa: E402
from lidar_object_detection.ops import nms_cuda  # noqa: E402

# Copied verbatim from lidar_object_detector_node.py. Importing that module pulls
# in rclpy + ros2_numpy, which this offline script does not need.
def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    keep = ((pts[:, 0] > point_range[0]) & (pts[:, 1] > point_range[1]) &
            (pts[:, 2] > point_range[2]) & (pts[:, 0] < point_range[3]) &
            (pts[:, 1] < point_range[4]) & (pts[:, 2] < point_range[5]))
    return pts[keep]


NCLASSES = 3


def cloud_to_xyzi(msg):
    """PointCloud2 -> (N,4) float32, matching the node's to_kitti_format path."""
    dtype = []
    for f in msg.fields:
        np_t = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}[f.datatype]
        dtype.append((f.name, np_t, f.offset))
    names = [d[0] for d in dtype]
    offsets = [d[2] for d in dtype]
    arr = np.frombuffer(msg.data, dtype={'names': names,
                                         'formats': [d[1] for d in dtype],
                                         'offsets': offsets,
                                         'itemsize': msg.point_step})
    i = arr['intensity'] if 'intensity' in names else np.zeros(len(arr), np.float32)
    pts = np.stack([arr['x'], arr['y'], arr['z'], i], -1).astype(np.float32)
    return pts[np.isfinite(pts).all(1)]


def bev_iou_axis_aligned(boxes):
    """Axis-aligned BEV IoU matrix -- what postprocess_kernels.cu's NMS uses."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)


def greedy_nms(boxes_xyxy, scores, thr):
    order = np.argsort(-scores)
    iou = bev_iou_axis_aligned(boxes_xyxy)
    keep, dead = [], np.zeros(len(scores), bool)
    for i in order:
        if dead[i]:
            continue
        keep.append(i)
        dead |= iou[i] > thr
        dead[i] = True
    return np.array(keep, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--weights', default=str(REF / 'weights' / 'epoch_160.pth'))
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--score-thr', type=float, default=0.1)
    ap.add_argument('--nms-thr', type=float, default=0.01)
    a = ap.parse_args()

    dev = torch.device('cuda')
    model = PointPillars(nclasses=NCLASSES).to(dev)
    model.load_state_dict(torch.load(a.weights))
    model.eval()

    NMS_PRE, MAX_NUM = model.nms_pre, model.max_num
    print(f"reference caps: nms_pre={NMS_PRE}  max_num={MAX_NUM}  "
          f"score_thr={a.score_thr}  nms_thr={a.nms_thr}\n")

    acc = {k: [] for k in ('anchors', 'old_topk', 'old_thr', 'old_nms', 'old_cap',
                           'new_thr', 'new_nms')}

    with AnyReader([Path(a.bag)]) as reader:
        conns = [c for c in reader.connections if 'PointCloud2' in c.msgtype]
        n = 0
        for conn, _, raw in reader.messages(connections=conns):
            if n >= a.frames:
                break
            pts = point_range_filter(cloud_to_xyzi(reader.deserialize(raw, conn.msgtype)))
            with torch.no_grad():
                # the model's own forward, stopped at the head (before decode)
                pillars, coors, npts = model.pillar_layer([torch.from_numpy(pts).to(dev)])
                feat = model.pillar_encoder(pillars, coors, npts)
                x = model.neck(model.backbone(feat))
                cls_p, box_p, dir_p = model.head(x)

            fms = torch.tensor(list(cls_p.size()[-2:]), device=dev)
            anchors = model.anchors_generator.get_multi_anchors(fms)

            c = cls_p[0].permute(1, 2, 0).reshape(-1, NCLASSES).sigmoid()
            b = box_p[0].permute(1, 2, 0).reshape(-1, 7)
            d = dir_p[0].permute(1, 2, 0).reshape(-1, 2)
            an = anchors.reshape(-1, 7)
            acc['anchors'].append(c.shape[0])

            # ---------------- OLD path (verbatim from get_predicted_bboxes_single)
            inds = c.max(1)[0].topk(NMS_PRE)[1]
            c_o, b_o, an_o = c[inds], b[inds], an[inds]
            acc['old_topk'].append(len(inds))
            bb = anchors2bboxes(an_o, b_o)
            xy, lw = bb[:, [0, 1]], bb[:, [3, 4]]
            bb2d = torch.cat([xy - lw / 2, xy + lw / 2, bb[:, 6:]], -1)
            n_thr = n_keep = 0
            for i in range(NCLASSES):
                s = c_o[:, i]
                m = s > a.score_thr
                if m.sum() == 0:
                    continue
                n_thr += int(m.sum())
                n_keep += len(nms_cuda(boxes=bb2d[m].contiguous(), scores=s[m].contiguous(),
                                       thresh=a.nms_thr, pre_maxsize=None, post_max_size=None))
            acc['old_thr'].append(n_thr)
            acc['old_nms'].append(n_keep)
            acc['old_cap'].append(min(n_keep, MAX_NUM))

            # ---------------- NEW path (postprocess_kernels.cu: argmax class, no caps)
            sc, cid = c.max(1)
            m = sc >= a.score_thr
            acc['new_thr'].append(int(m.sum()))
            bb_n = anchors2bboxes(an[m], b[m])
            xy, lw = bb_n[:, [0, 1]], bb_n[:, [3, 4]]
            nb = torch.cat([xy - lw / 2, xy + lw / 2], -1).cpu().numpy()
            ns, ncid = sc[m].cpu().numpy(), cid[m].cpu().numpy()
            tot = 0
            for i in range(NCLASSES):
                sel = ncid == i
                if sel.sum():
                    tot += len(greedy_nms(nb[sel], ns[sel], a.nms_thr))
            acc['new_nms'].append(tot)

            n += 1
            print(f"  frame {n:2d}  anchors={acc['anchors'][-1]}  "
                  f"OLD topk={acc['old_topk'][-1]} thr={n_thr} nms={n_keep} "
                  f"cap={acc['old_cap'][-1]}   "
                  f"NEW thr={acc['new_thr'][-1]} nms={acc['new_nms'][-1]}", flush=True)

    print("\n=== mean per frame ===")
    for k in ('anchors', 'old_topk', 'old_thr', 'old_nms', 'old_cap', 'new_thr', 'new_nms'):
        print(f"  {k:10s} {np.mean(acc[k]):9.1f}")
    print(f"\nFINAL   OLD {np.mean(acc['old_cap']):.1f}   NEW {np.mean(acc['new_nms']):.1f}")


if __name__ == '__main__':
    main()
