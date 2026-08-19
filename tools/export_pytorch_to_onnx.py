#!/usr/bin/env python3
"""Export the repo's trained PyTorch PointPillars (epoch_160.pth) to the ONNX
graph that CUDA-PointPillars consumes.

The ONNX deliberately starts AFTER voxelisation: the CUDA pipeline does
voxelisation and pillar-feature construction itself, so the graph takes the
already-built pillar features and runs
    conv1d -> BN -> ReLU -> max-pool -> PPScatterPlugin -> backbone -> neck -> head

Two things make this model different from a stock OpenPCDet one, and both are
handled here rather than in CUDA:

  * It is the zhulf0804/mmdet3d-style implementation, whose PillarEncoder takes
    NINE pillar features, not OpenPCDet's ten. Those nine are a pure permutation
    of values the CUDA kernel already computes, so preprocess_kernels.cu emits
    them directly when PILLAR_FEATURE_LAYOUT is mmdet3d9.
  * Its class order is Pedestrian, Cyclist, Car -- the same order as
    object_detection_msgs, so no class remap is needed at runtime.

Usage:
    python3 tools/export_pytorch_to_onnx.py \
        --weights reference/lidar_object_detection_py/weights/epoch_160.pth \
        --output  src/cuda_pointpillars_ros/model/pointpillar_gp.onnx
"""
import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "reference" / "lidar_object_detection_py"


def import_model_module():
    """Import the model definitions without the compiled CUDA extensions.

    pointpillars.py pulls in `..ops` (Voxelization, nms_cuda) and `..utils` at
    module scope. Those need compiled extensions that are not built here, and
    none of them are used by the layers we export -- voxelisation and NMS both
    live in CUDA on the deployment side. Stub them so the import succeeds.
    """
    sys.path.insert(0, str(REF))

    class _Stub(types.ModuleType):
        """Any attribute resolves to a no-op, so whatever these modules are
        asked for at import time succeeds. Nothing we export calls into them."""

        def __getattr__(self, name):
            # Dunders must fail normally: torch's exporter runs inspect over
            # loaded modules, and resolving __file__ to a lambda breaks it.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return lambda *a, **k: None

    for name in ("lidar_object_detection.ops", "lidar_object_detection.utils"):
        sys.modules[name] = _Stub(name)

    from lidar_object_detection.model import pointpillars as pp  # noqa: E402
    return pp


class PPScatter(torch.autograd.Function):
    """Emits the PPScatterPlugin node that the TensorRT runtime provides.

    TensorRT matches the ONNX op_type against a registered plugin name, and
    turns the node's attributes into plugin fields -- so `dense_shape` here
    becomes the (grid_y, grid_x) the plugin scatters into.
    """

    @staticmethod
    def symbolic(g, features, coords, params, dense_h, dense_w):
        return g.op(
            "PPScatterPlugin", features, coords, params,
            dense_shape_i=[dense_h, dense_w])

    @staticmethod
    def forward(ctx, features, coords, params, dense_h, dense_w):
        # Shape-only stub; the real scatter happens in the plugin at runtime.
        return torch.zeros(
            1, features.shape[1], dense_h, dense_w,
            dtype=features.dtype, device=features.device)


class ExportGraph(nn.Module):
    def __init__(self, pp_mod, state, nclasses, grid_h, grid_w, in_channel):
        super().__init__()
        enc = pp_mod.PillarEncoder(
            voxel_size=[0.16, 0.16, 4], point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            in_channel=in_channel, out_channel=64)
        self.conv, self.bn = enc.conv, enc.bn
        self.backbone = pp_mod.Backbone(in_channel=64, out_channels=[64, 128, 256],
                                        layer_nums=[3, 5, 5])
        self.neck = pp_mod.Neck(in_channels=[64, 128, 256], upsample_strides=[1, 2, 4],
                                out_channels=[128, 128, 128])
        self.head = pp_mod.Head(in_channel=384, n_anchors=2 * nclasses, n_classes=nclasses)
        self.grid_h, self.grid_w = grid_h, grid_w

        def take(prefix, module):
            sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            missing = module.load_state_dict(sub, strict=True)
            return len(sub)

        n = 0
        n += take("pillar_encoder.conv.", self.conv)
        n += take("pillar_encoder.bn.", self.bn)
        n += take("backbone.", self.backbone)
        n += take("neck.", self.neck)
        n += take("head.", self.head)
        print(f"loaded {n} tensors from the checkpoint")

    def forward(self, voxels, voxel_idxs, voxel_num):
        # voxels: (P, max_points, C) -- already masked to zero for padded points
        # by the CUDA kernel, exactly as PillarEncoder's own mask would.
        x = voxels.permute(0, 2, 1)                    # (P, C, max_points)
        x = F.relu(self.bn(self.conv(x)))              # (P, 64, max_points)
        x = torch.max(x, dim=-1)[0]                    # (P, 64)
        canvas = PPScatter.apply(x, voxel_idxs, voxel_num, self.grid_h, self.grid_w)
        feats = self.neck(self.backbone(canvas))
        cls_preds, box_preds, dir_preds = self.head(feats)
        # postprocess_kernels.cu indexes these as NHWC.
        return (cls_preds.permute(0, 2, 3, 1).contiguous(),
                box_preds.permute(0, 2, 3, 1).contiguous(),
                dir_preds.permute(0, 2, 3, 1).contiguous())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(REF / "weights" / "epoch_160.pth"))
    ap.add_argument("--output", default=str(REPO / "src/cuda_pointpillars_ros/model/pointpillar_gp.onnx"))
    ap.add_argument("--nclasses", type=int, default=3)
    ap.add_argument("--max-voxels", type=int, default=40000)
    ap.add_argument("--max-points", type=int, default=32)
    ap.add_argument("--features", type=int, default=9,
                    help="9 for this mmdet3d-style model; 10 for OpenPCDet")
    ap.add_argument("--grid", type=int, nargs=2, default=[496, 432],
                    metavar=("GRID_Y", "GRID_X"))
    ap.add_argument("--opset", type=int, default=11)
    args = ap.parse_args()

    pp_mod = import_model_module()
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    if "model_state" in state:
        state = state["model_state"]

    model = ExportGraph(pp_mod, state, args.nclasses, args.grid[0], args.grid[1],
                        args.features).eval()

    voxels = torch.zeros(args.max_voxels, args.max_points, args.features)
    voxel_idxs = torch.zeros(args.max_voxels, 4, dtype=torch.int32)
    voxel_num = torch.zeros(1, dtype=torch.int32)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (voxels, voxel_idxs, voxel_num), args.output,
        input_names=["voxels", "voxel_idxs", "voxel_num"],
        output_names=["cls_preds", "box_preds", "dir_cls_preds"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False)
    print(f"wrote {args.output}")

    try:
        import onnx
        m = onnx.load(args.output)
        print("inputs :", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                           for i in m.graph.input])
        print("outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                           for o in m.graph.output])
        ops = {n.op_type for n in m.graph.node}
        print("PPScatterPlugin present:", "PPScatterPlugin" in ops)
    except ImportError:
        print("(install onnx to verify the graph)")


if __name__ == "__main__":
    main()
