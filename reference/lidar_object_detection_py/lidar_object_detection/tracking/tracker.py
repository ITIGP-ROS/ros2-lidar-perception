# Thin wrapper around the vendored AB3DMOT tracker, exposing a single-frame
# update() API for the PointPillars output format:
#   lidar_bboxes (N, 7): [x, y, z, w, l, h, yaw]   (z = bottom center)
#   scores (N,), labels (N,)
# One AB3DMOT instance per object class (KITTI label: 0 pedestrian, 1 cyclist,
# 2 car); all classes share a global ID counter so IDs never collide.
import numpy as np

from .model import AB3DMOT

# KITTI label -> tracker category name
_LABEL_TO_CAT = {
    0: 'Pedestrian',
    1: 'Cyclist',
    2: 'Car',
}

# output column layout: [x, y, z, w, l, h, yaw, score, label, track_id]
OUT_COLS = 10
_RAW2LIDAR_IDX = [3, 4, 5, 1, 2, 0, 6]   # [h,w,l,x,y,z,theta] -> [x,y,z,w,l,h,theta]
_LIDAR2RAW_IDX = [5, 3, 4, 0, 1, 2, 6]   # [x,y,z,w,l,h,theta] -> [h,w,l,x,y,z,theta]


class Track3D(object):

    def __init__(self):
        self._trackers = {}
        self._next_id = 1
        self._frame = 0

    def _get_tracker(self, label):
        if label not in _LABEL_TO_CAT:
            return None
        if label not in self._trackers:
            self._trackers[label] = AB3DMOT(_LABEL_TO_CAT[label], id_allocator=self._alloc_id)
        return self._trackers[label]

    def _alloc_id(self):
        self._next_id += 1
        return self._next_id - 1

    def update(self, lidar_bboxes, scores, labels):
        """Run one frame of tracking.

        Args:
            lidar_bboxes: (N, 7) float array [x, y, z, w, l, h, yaw]
            scores: (N,) float array of detection confidences
            labels: (N,) int array of class labels (0, 1, 2)

        Returns:
            (M, 10) float array [x, y, z, w, l, h, yaw, score, label, track_id]
            M may differ from N (tracks survive missed detections).
        """
        lidar_bboxes = np.asarray(lidar_bboxes, dtype=np.float32).reshape(-1, 7)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        if lidar_bboxes.shape[0] == 0:
            lidar_bboxes = np.empty((0, 7), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)

        self._frame += 1

        # instantiate trackers for labels seen in this frame before processing
        for label in np.unique(labels):
            self._get_tracker(int(label))

        out_list = []
        # a tracker instance must be advanced every frame even with no detections
        for label, tracker in self._trackers.items():
            sel = np.where(labels == label)[0]
            if len(sel):
                dets_raw = lidar_bboxes[sel][:, _LIDAR2RAW_IDX]
                info = np.column_stack([scores[sel], np.full(len(sel), float(label))])
            else:
                dets_raw = np.empty((0, 7), dtype=np.float32)
                info = np.empty((0, 2), dtype=np.float32)
            dets_all = {'dets': dets_raw, 'info': info}
            results = tracker.track(dets_all, self._frame)
            if results.shape[0] > 0:
                out_list.append(self._to_output(results))

        if len(out_list) == 0:
            return np.empty((0, OUT_COLS), dtype=np.float32)
        return np.vstack(out_list)

    def _to_output(self, results):
        # results: (M, 7 + 1 + 2) = [h,w,l,x,y,z,theta, id, score, label]
        box_lidar = results[:, :7][:, _RAW2LIDAR_IDX]
        ids = results[:, 7:8]
        scores = results[:, 8:9]
        labels = results[:, 9:10]
        return np.hstack([box_lidar, scores, labels, ids])

    def reset(self):
        self._trackers = {}
        self._next_id = 1
        self._frame = 0
