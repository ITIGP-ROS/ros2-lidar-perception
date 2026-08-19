#!/usr/bin/env python3
"""Replay PointCloud2 messages from a rosbag2/MCAP bag onto a live topic.

A stand-in for `ros2 bag play`, which is not installed on every machine here
(`ros2` on this workstation only provides control/daemon/node/param/service).
Reads the bag with `rosbags`, which needs no ROS bag tooling at all.

    python3 tools/play_bag.py --bag <dir> --topic /kitti/velo --rate 3

`--lockstep` publishes the next cloud only once the pipeline has answered for
the previous one. Prefer it for correctness runs: the node's input QoS is
depth=1, so free-running publication silently drops frames whenever inference
is slower than the publish rate.
"""
import argparse
import numpy as np
import rclpy
from pathlib import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from rosbags.highlevel import AnyReader

QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                 history=QoSHistoryPolicy.KEEP_LAST,
                 durability=QoSDurabilityPolicy.VOLATILE, depth=1)
LABELS = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}


def to_ros(src):
    m = PointCloud2()
    m.header.frame_id = src.header.frame_id
    m.height, m.width = int(src.height), int(src.width)
    m.fields = [PointField(name=f.name, offset=int(f.offset),
                           datatype=int(f.datatype), count=int(f.count)) for f in src.fields]
    m.is_bigendian = bool(src.is_bigendian)
    m.point_step = int(src.point_step)
    m.row_step = int(src.row_step)
    m.data = src.data.tobytes()
    m.is_dense = bool(src.is_dense)
    return m


class Player(Node):
    def __init__(self, args):
        super().__init__('bag_player')
        self.a = args
        self.pub = self.create_publisher(PointCloud2, args.topic, QOS)
        self.sent = 0
        self.got = 0
        self.saved = []
        self.ids = set()
        self.reader = AnyReader([Path(args.bag)])
        self.reader.open()
        conns = [c for c in self.reader.connections if c.topic == args.topic]
        if not conns:
            avail = sorted({c.topic for c in self.reader.connections})
            raise SystemExit(f"topic {args.topic!r} not in bag. Available: {avail}")
        self.it = self.reader.messages(connections=conns)

        if args.watch:
            from object_detection_msgs.msg import Object3dArray
            self.create_subscription(Object3dArray, args.watch, self.on_out, QOS)

        if args.lockstep:
            self.create_timer(2.0, self.watchdog)   # recover if a reply is lost
            self.send()
        else:
            self.create_timer(1.0 / args.rate, self.send)

    def send(self):
        if self.a.count and self.sent >= self.a.count:
            return
        try:
            conn, ts, raw = next(self.it)
        except StopIteration:
            self.a.count = self.sent
            return
        m = to_ros(self.reader.deserialize(raw, conn.msgtype))
        m.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(m)
        self.sent += 1
        if not self.a.watch and self.sent % 20 == 0:
            print(f"  published {self.sent}", flush=True)

    def watchdog(self):
        if self.sent > self.got:
            self.send()

    def on_out(self, msg):
        self.got += 1
        rows = []
        for o in msg.objects:
            c = np.array([[p.x, p.y, p.z] for p in o.bounding_box.corners])
            ctr = c.mean(0)
            self.ids.add(o.track_id)
            rows.append(f"{o.track_id}:{LABELS.get(o.label, o.label)[:3]}"
                        f"({ctr[0]:.1f},{ctr[1]:.1f})s{o.confidence_score:.2f}")
        if not self.a.save:
            print(f"[{self.got:4d}] n={len(rows):2d}  " + " ".join(rows[:6]), flush=True)
        else:
            # Save z as well as x/y. Leaving it out once hid a real bug: the
            # boxes were being published h/2 too high and nothing here noticed.
            def centroid(o):
                return np.array([[p.x, p.y, p.z] for p in o.bounding_box.corners]).mean(0)
            self.saved.append([dict(zip(('x', 'y', 'z'), map(float, centroid(o))),
                                    score=float(o.confidence_score), label=int(o.label),
                                    tid=int(o.track_id)) for o in msg.objects])
            if self.got % 10 == 0: print(f"  {self.got}", flush=True)
        if self.a.lockstep:
            self.send()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--topic', default='/kitti/velo')
    ap.add_argument('--rate', type=float, default=3.0, help='Hz (ignored with --lockstep)')
    ap.add_argument('--count', type=int, default=0, help='0 = whole bag')
    ap.add_argument('--lockstep', action='store_true')
    ap.add_argument('--watch', default='', metavar='TOPIC',
                    help='subscribe to an Object3dArray topic and print what comes back')
    ap.add_argument('--save', default='', metavar='FILE',
                    help='also write the received detections to JSON, for comparing runs')
    a = ap.parse_args()

    rclpy.init()
    n = Player(a)
    idle = 0
    try:
        while rclpy.ok():
            before = (n.sent, n.got)
            rclpy.spin_once(n, timeout_sec=0.5)
            done = a.count and n.sent >= a.count and (not a.watch or n.got >= a.count)
            idle = idle + 1 if (n.sent, n.got) == before else 0
            if done or idle > 60:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if a.watch:
            print(f"\nSUMMARY: {n.sent} published, {n.got} answered, "
                  f"{len(n.ids)} distinct track ids")
        if a.save:
            import json
            json.dump(n.saved, open(a.save, 'w'))
            print(f"wrote {a.save}: {len(n.saved)} frames")
        rclpy.shutdown()


if __name__ == '__main__':
    main()
