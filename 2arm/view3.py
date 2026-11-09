#!/usr/bin/env python
"""view3.py — 三路相机同时拍：左腕 /dev/video2、右腕 /dev/video0、头部 /dev/video4。

用法：
  python view3.py --tag base                     # 不动臂，只拍三张
  python view3.py --arm left --goto table_hover --tag hover
  python view3.py --arm left --pose "-48.2,-36.1,49.0,51.4,3.3" --tag grasp

图片存到 /tmp/three_<tag>_left.jpg / _right.jpg / _head.jpg，
同时在终端打印每路检测到的方块像素。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import armcore as C
from armdriver import DualArm
from armvision import CameraStream, Detector

CAMS = {"left": "/dev/video2", "right": "/dev/video0", "head": "/dev/video4"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["left", "right"], default="left")
    ap.add_argument("--goto", default="", help="移动到某个点位名")
    ap.add_argument("--pose", default="", help="移动到指定 5 个关节值，逗号分隔")
    ap.add_argument("--tag", default="shot")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="view3")
    stop = threading.Event()
    dual = DualArm(cfg, logger, stop)
    det = Detector(cfg, logger)
    cams: dict[str, CameraStream] = {}
    for name, dev in CAMS.items():
        cam = CameraStream(dev, logger=logger, name=name, width=640, height=480, fps=30)
        cam.start()
        cams[name] = cam

    try:
        time.sleep(1.2)
        target = None
        if args.pose:
            target = np.array([float(v) for v in args.pose.split(",")], dtype=float)
        elif args.goto:
            target = np.asarray(C.get_pose(cfg, args.arm, args.goto), dtype=float)
        if target is not None:
            arm = dual.arms[args.arm]
            arm.bus.connect()
            arm.bus.enable_torque()
            arm.set_gripper(float(cfg["arms"][args.arm]["gripper_open"]), settle=0.4)
            arm.start([np.clip(target, -100, 100)],
                      gripper=float(cfg["arms"][args.arm]["gripper_open"]),
                      label="view3", speed_scale=0.25)
            ok = arm.wait(timeout=12.0)
            time.sleep(0.4)
            q = arm.bus.read_joints()
            print(f"[{args.arm}] 到位={ok} 残差={np.abs(q - target).max():.2f} "
                  f"q={np.round(q, 1).tolist()}")
            arm.bus.disable_torque()
            arm.bus.disconnect()

        for name, cam in cams.items():
            f = cam.read(timeout=1.5)
            if f is None:
                print(f"  {name}: 没画面")
                continue
            d = det.detect(f, force_yolo=True)
            print(f"  {name}: {d}")
            img = det.annotate(f.copy(), d) if d is not None else f.copy()
            cv2.imwrite(f"/tmp/three_{args.tag}_{name}.jpg", img)
        print(f"已存到 /tmp/three_{args.tag}_*.jpg")
    finally:
        for cam in cams.values():
            try:
                cam.stop()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
