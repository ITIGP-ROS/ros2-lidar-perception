# Adapted from xinshuoweng/AB3DMOT (MIT License), AB3DMOT_libs/model.py
# https://github.com/xinshuoweng/AB3DMOT
# A Baseline of 3D Multi-Object Tracking: Kalman filter prediction + Hungarian /
# greedy association on 3D/GIoU affinity, per object class. Runs directly in the
# LIDAR frame (z up); ego-motion compensation and visualization are stripped out.
import numpy as np

from .box import Box3D
from .matching import data_association
from .kalman_filter import KF

np.set_printoptions(suppress=True, precision=3)

# matching parameters tuned for KITTI-style lidar detections (PV-RCNN values from AB3DMOT)
# metric thresholds are NEGATIVE because the affinity is a cost to be minimized
_PARAMS = {
    'Car':         dict(algm='hungar', metric='giou_3d', thres=-0.2, min_hits=3, max_age=2),
    'Pedestrian':  dict(algm='greedy', metric='giou_3d', thres=-0.4, min_hits=1, max_age=4),
    'Cyclist':     dict(algm='hungar', metric='dist_3d', thres=2.0, min_hits=3, max_age=4),
}


class AB3DMOT(object):

    def __init__(self, cat, id_allocator=None):
        # counter
        self.trackers = []
        self.frame_count = 0
        self.id_now_output = []

        # id_allocator: optional callable returning globally unique track IDs
        # (shared across per-class trackers); defaults to a private counter
        self.id_allocator = id_allocator or self._default_id

        # config
        self.cat = cat
        p = _PARAMS[cat]
        self.algm, self.metric, self.thres, self.max_age, self.min_hits = \
            p['algm'], p['metric'], p['thres'], p['max_age'], p['min_hits']

        # define max/min values for the output affinity matrix
        if self.metric in ['dist_3d', 'dist_2d', 'm_dis']:
            self.max_sim, self.min_sim = 0.0, -100.
        elif self.metric in ['iou_2d', 'iou_3d']:
            self.max_sim, self.min_sim = 1.0, 0.0
        elif self.metric in ['giou_2d', 'giou_3d']:
            self.max_sim, self.min_sim = 1.0, -1.0
        self.ID_count = [0]

    def process_dets(self, dets):
        # convert each detection into the class Box3D
        # dets: numpy array in the format [[h, w, l, x, y, z, theta], ...]
        dets_new = []
        for det in dets:
            det_tmp = Box3D.array2bbox_raw(det)
            dets_new.append(det_tmp)
        return dets_new

    def within_range(self, theta):
        # make sure the orientation is within a proper range
        if theta >= np.pi:
            theta -= np.pi * 2
        if theta < -np.pi:
            theta += np.pi * 2
        return theta

    def orientation_correction(self, theta_pre, theta_obs):
        # update orientation in propagated tracks and detected boxes so that they are within 90 degree

        theta_pre = self.within_range(theta_pre)
        theta_obs = self.within_range(theta_obs)

        # if the angle of two theta is not acute angle, then make it acute
        if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:
            theta_pre += np.pi
            theta_pre = self.within_range(theta_pre)

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
            if theta_obs > 0:
                theta_pre += np.pi * 2
            else:
                theta_pre -= np.pi * 2

        return theta_pre, theta_obs

    def prediction(self):
        # get predicted locations from existing tracks

        trks = []
        for t in range(len(self.trackers)):
            # propagate locations
            kf_tmp = self.trackers[t]
            kf_tmp.kf.predict()
            kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])

            # update statistics
            kf_tmp.time_since_update += 1
            trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
            trks.append(Box3D.array2bbox(trk_tmp))

        return trks

    def update(self, matched, unmatched_trks, dets, info):
        # update matched trackers with assigned detections

        dets = list(dets)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
                assert len(d) == 1, 'error'

                # update statistics
                trk.time_since_update = 0        # reset because just updated
                trk.hits += 1

                # update orientation in propagated tracks and detected boxes so that they are within 90 degree
                bbox3d = Box3D.bbox2array(dets[d[0]])
                trk.kf.x[3], bbox3d[3] = self.orientation_correction(trk.kf.x[3], bbox3d[3])

                # kalman filter update with observation
                trk.kf.update(bbox3d)

                trk.kf.x[3] = self.within_range(trk.kf.x[3])
                trk.info = info[d, :][0]

    def _default_id(self):
        self.ID_count[0] += 1
        return self.ID_count[0] - 1

    def birth(self, dets, info, unmatched_dets):
        # create and initialise new trackers for unmatched detections

        new_id_list = list()                    # new ID generated for unmatched detections
        for i in unmatched_dets:                # a scalar of index
            trk = KF(Box3D.bbox2array(dets[i]), info[i, :], self.id_allocator())
            self.trackers.append(trk)
            new_id_list.append(trk.id)

        return new_id_list

    def output(self):
        # output exiting tracks that have been stably associated, i.e., >= min_hits
        # and also delete tracks that have appeared for a long time, i.e., >= max_age

        num_trks = len(self.trackers)
        results = []
        for trk in reversed(self.trackers):
            # change format from [x, y, z, theta, l, w, h] to [h, w, l, x, y, z, theta]
            d = Box3D.array2bbox(trk.kf.x[:7].reshape((7, )))     # bbox location self
            d = Box3D.bbox2array_raw(d)

            if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                results.append(np.concatenate((d, [trk.id], trk.info)).reshape(1, -1))
            num_trks -= 1

            # death, remove dead tracklet
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(num_trks)

        return results

    def track(self, dets_all, frame):
        """
        Params:
            dets_all: dict
                dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
                info: an array of other info for each det (score, label, ...)
            frame: int, frame number (only used for bookkeeping)
        Requires: this method must be called once for each frame even with empty detections.
        Returns a numpy array [N, 7 + 1 + info_cols]: [h,w,l,x,y,z,theta, ID, info...]
        """
        dets, info = dets_all['dets'], dets_all['info']
        self.frame_count += 1

        # process detection format
        dets = self.process_dets(dets)

        # tracks propagation based on velocity
        trks = self.prediction()

        # matching
        trk_innovation_matrix = None
        if self.metric == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers]
        matched, unmatched_dets, unmatched_trks, _, _ = \
            data_association(dets, trks, self.metric, self.thres, self.algm, trk_innovation_matrix)

        # update trks with matched detection measurement
        self.update(matched, unmatched_trks, dets, info)

        # create and initialise new trackers for unmatched detections
        self.birth(dets, info, unmatched_dets)

        # output existing valid tracks
        results = self.output()
        if len(results) > 0:
            results = np.concatenate(results)
        else:
            results = np.empty((0, 7 + 1 + info.shape[1]))

        return results
