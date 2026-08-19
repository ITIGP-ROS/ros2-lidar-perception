import numpy as np

def rotz(theta):
    """Rotation matrix about the z axis (lidar frame, z up)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])

class Box3D:
    """3D bounding box in the LIDAR frame.

    x: forward, y: left, z: up. The stored (x, y, z) is the BOTTOM CENTER
    of the box (ground level), matching the PointPillars lidar convention.
    """

    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, ry=None, s=None):
        self.x = x
        self.y = y
        self.z = z
        self.h = h
        self.w = w
        self.l = l
        self.ry = ry
        self.s = s
        self.corners_3d = None

    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.ry, self.l, self.w, self.h, self.s)

    @classmethod
    def bbox2array(cls, bbox):
        # [x, y, z, ry, l, w, h] (center form, used by the Kalman filter state)
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h, bbox.s])

    @classmethod
    def bbox2array_raw(cls, bbox):
        # [h, w, l, x, y, z, ry] (KITTI raw form)
        if bbox.s is None:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry])
        else:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry, bbox.s])

    @classmethod
    def array2bbox_raw(cls, data):
        # take the format of data of [h, w, l, x, y, z, theta]
        bbox = Box3D()
        bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox

    @classmethod
    def array2bbox(cls, data):
        # take the format of data of [x, y, z, theta, l, w, h]
        bbox = Box3D()
        bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox

    @classmethod
    def box2corners3d(cls, bbox):
        """Convert a box (bottom center at x,y,z) to its 8 corners in the lidar frame.

        Corner order (0-3 top, 4-7 bottom, bottom CCW order 7,6,5,4):
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        """
        if bbox.corners_3d is not None:
            return bbox.corners_3d

        R = rotz(bbox.ry)
        l, w, h = bbox.l, bbox.w, bbox.h

        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [h, h, h, h, 0, 0, 0, 0]

        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += bbox.x
        corners_3d[1, :] += bbox.y
        corners_3d[2, :] += bbox.z
        corners_3d = np.transpose(corners_3d)
        bbox.corners_3d = corners_3d

        return corners_3d
