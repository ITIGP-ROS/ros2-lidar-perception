"""Generate golden vectors from the reference Python AB3DMOT implementation.

These pin the C++ port in src/lidar_tracking. Run from the repo root:
    python3 gen_goldens.py <repo_root>

Nothing here re-implements the reference: bbox3d2corners is lifted verbatim out
of utils/process.py via the AST (that file imports torch/numba, so it cannot be
imported directly), and everything else is imported from the real package.
"""
import ast
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.')
REF = os.path.join(REPO, 'reference', 'lidar_object_detection_py')
OUT = os.path.join(REPO, 'src', 'lidar_tracking', 'test', 'data')
sys.path.insert(0, REF)

from lidar_object_detection.tracking import Track3D                     # noqa: E402
from lidar_object_detection.tracking.box import Box3D                   # noqa: E402
from lidar_object_detection.tracking.model import AB3DMOT               # noqa: E402
from lidar_object_detection.tracking.kalman_filter import KF            # noqa: E402
from lidar_object_detection.tracking.dist_metrics import iou, dist3d    # noqa: E402


def load_bbox3d2corners():
    """Lift bbox3d2corners verbatim from utils/process.py (which imports torch)."""
    path = os.path.join(REF, 'lidar_object_detection', 'utils', 'process.py')
    src = open(path).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'bbox3d2corners':
            ns = {'np': np}
            exec(compile(ast.Module([node], []), path, 'exec'), ns)
            return ns['bbox3d2corners']
    raise RuntimeError('bbox3d2corners not found in process.py')


bbox3d2corners = load_bbox3d2corners()
rng = np.random.RandomState(0xAB3D)
os.makedirs(OUT, exist_ok=True)


def dump(name, obj):
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(obj, f, indent=1)
    print(f'  {name}')


def rand_box(rng, big=True):
    """[x, y, z, w, l, h, yaw] with z = BOTTOM centre."""
    dims = [3.9, 1.6, 1.56] if big else [0.8, 0.6, 1.73]
    return [
        float(rng.uniform(-30, 30)), float(rng.uniform(-20, 20)),
        float(rng.uniform(-2.0, -0.5)),
        dims[0] * float(rng.uniform(0.8, 1.2)),
        dims[1] * float(rng.uniform(0.8, 1.2)),
        dims[2] * float(rng.uniform(0.8, 1.2)),
        float(rng.uniform(-np.pi, np.pi)),
    ]


# ── 1. bbox3d2corners: the IVI-facing pose code ───────────────────────────────
boxes = [rand_box(rng, i % 2 == 0) for i in range(24)]
boxes += [                       # axis-aligned cases, easy to eyeball
    [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
    [1.0, 2.0, -1.0, 4.0, 2.0, 1.5, np.pi / 2],
    [-5.0, 3.0, -1.7, 3.9, 1.6, 1.56, np.pi],
    [2.0, -2.0, -1.7, 3.9, 1.6, 1.56, -np.pi / 4],
]
dump('corners.json', [
    {'box': b, 'corners': bbox3d2corners(np.array([b], dtype=np.float32))[0].tolist()}
    for b in boxes
])

# ── 2. giou_3d / iou_3d / dist3d ──────────────────────────────────────────────
pairs = []
for i in range(24):
    a = rand_box(rng, i % 2 == 0)
    # half the pairs deliberately overlap, half are far apart
    if i % 2 == 0:
        b = list(a)
        b[0] += float(rng.uniform(-2.5, 2.5))
        b[1] += float(rng.uniform(-1.5, 1.5))
        b[6] += float(rng.uniform(-0.6, 0.6))
    else:
        b = rand_box(rng, i % 2 == 0)
    ba, bb = Box3D.array2bbox(np.array(a)[[0, 1, 2, 6, 4, 3, 5]]), None
    # Box3D.array2bbox takes [x, y, z, theta, l, w, h]
    ba = Box3D.array2bbox(np.array([a[0], a[1], a[2], a[6], a[4], a[3], a[5]]))
    bb = Box3D.array2bbox(np.array([b[0], b[1], b[2], b[6], b[4], b[3], b[5]]))
    pairs.append({
        'a_xyzwlhy': a, 'b_xyzwlhy': b,
        'giou_3d': float(iou(ba, bb, 'giou_3d')),
        'iou_3d': float(iou(ba, bb, 'iou_3d')),
        'dist_3d': float(dist3d(ba, bb)),
    })
dump('metrics.json', pairs)

# ── 3. orientation_correction / within_range branch boundaries ────────────────
probe = AB3DMOT('Car')
grid = [-2 * np.pi, -np.pi * 1.5, -np.pi, -np.pi * 0.75, -np.pi / 2, -0.3, 0.0,
        0.3, np.pi / 2, np.pi * 0.75, np.pi, np.pi * 1.5, 2 * np.pi, 3.0, -3.0]
orient = []
for tp in grid:
    for to in grid:
        pre, obs = probe.orientation_correction(float(tp), float(to))
        orient.append({'theta_pre': float(tp), 'theta_obs': float(to),
                       'out_pre': float(pre), 'out_obs': float(obs)})
dump('orientation.json', {
    'within_range': [{'in': float(t), 'out': float(probe.within_range(float(t)))}
                     for t in grid],
    'orientation_correction': orient,
})

# ── 4. Kalman filter: exact filterpy defaults, Joseph-form update ─────────────
init = np.array([10.0, 2.0, -1.7, 0.1, 3.9, 1.6, 1.56])   # [x,y,z,theta,l,w,h]
kf = KF(init.copy(), np.array([0.9, 2.0]), 7)
steps = [{'step': 'init', 'x': kf.kf.x.reshape(-1).tolist(),
          'P_diag': np.diag(kf.kf.P).tolist()}]
meas = [
    [10.5, 2.1, -1.70, 0.12, 3.9, 1.6, 1.56],
    [11.1, 2.0, -1.71, 0.09, 3.9, 1.6, 1.56],
    [11.6, 1.9, -1.69, 0.11, 3.9, 1.6, 1.56],
]
for i, m in enumerate(meas):
    kf.kf.predict()
    steps.append({'step': f'predict{i}', 'x': kf.kf.x.reshape(-1).tolist(),
                  'P_diag': np.diag(kf.kf.P).tolist()})
    kf.kf.update(np.array(m))
    steps.append({'step': f'update{i}', 'z': m, 'x': kf.kf.x.reshape(-1).tolist(),
                  'P_diag': np.diag(kf.kf.P).tolist()})
dump('kalman.json', {
    'init_state_xyz_theta_lwh': init.tolist(),
    'P0_diag': (np.eye(10)[0] * 0).tolist(),   # placeholder, real values in steps
    'steps': steps,
})

# ── 5. Full Track3D scenarios ─────────────────────────────────────────────────
CAR, PED = 2, 0     # this repo's convention: Pedestrian 0, Cyclist 1, Car 2


def run(frames):
    t = Track3D()
    out = []
    for boxes, scores, labels in frames:
        r = t.update(np.array(boxes, dtype=np.float32).reshape(-1, 7),
                     np.array(scores, dtype=np.float32).reshape(-1),
                     np.array(labels, dtype=np.int64).reshape(-1))
        out.append(np.asarray(r, dtype=np.float64).tolist())
    return out


# (a) one car driving away in a straight line
frames_a = []
for k in range(10):
    frames_a.append(([[10.0 + 0.5 * k, 0.0 + 0.02 * k, -1.7, 3.9, 1.6, 1.56, 0.01 * k]],
                     [0.9], [CAR]))

# (b) two pedestrians crossing (exercises association)
frames_b = []
for k in range(12):
    frames_b.append(([
        [8.0, -3.0 + 0.5 * k, -1.7, 0.8, 0.6, 1.73, 1.57],
        [8.3, 3.0 - 0.5 * k, -1.7, 0.8, 0.6, 1.73, -1.57],
    ], [0.85, 0.8], [PED, PED]))

# (c) a car that disappears for 3 frames then returns (max_age / birth-death)
frames_c = []
for k in range(12):
    if 5 <= k <= 7:
        frames_c.append(([], [], []))
    else:
        frames_c.append(([[12.0 + 0.4 * k, 1.0, -1.7, 3.9, 1.6, 1.56, 0.0]],
                         [0.88], [CAR]))

# (d) a car whose detected heading flips by ~pi between frames. This is what
#     the PointPillars direction classifier does in practice, and it is the only
#     scenario that exercises orientation_correction inside the KF update.
frames_d = []
for k in range(10):
    yaw = 0.05 if k % 2 == 0 else 0.05 + np.pi
    frames_d.append(([[9.0 + 0.45 * k, -1.0, -1.7, 3.9, 1.6, 1.56, yaw]],
                     [0.87], [CAR]))

dump('sequences.json', {
    'columns': 'x,y,z,w,l,h,yaw,score,label,track_id',
    'note': 'label uses THIS repo order: Pedestrian=0, Cyclist=1, Car=2',
    'straight_line_car': {'input': [[b, s, l] for b, s, l in frames_a],
                          'output': run(frames_a)},
    'crossing_pedestrians': {'input': [[b, s, l] for b, s, l in frames_b],
                             'output': run(frames_b)},
    'dropout_and_return': {'input': [[b, s, l] for b, s, l in frames_c],
                           'output': run(frames_c)},
    'yaw_flip': {'input': [[b, s, l] for b, s, l in frames_d],
                 'output': run(frames_d)},
})

print('\ngoldens written to', OUT)
