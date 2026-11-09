#!/usr/bin/env python
"""peek_box.py — 只观察不动爪：把右臂移到 box_hover，拍带检测框的图后回 home。

诊断用：确认盒内方块在腕相机里的真实像素位置，以及检测是否被盒壁反光误导。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import armcore as C
from dualarm_loop import Supervisor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["left", "right"], default="right")
    ap.add_argument("--tag", default="peek_box")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="peek_box")
    sup = Supervisor(cfg, logger)
    sup._signals()
    try:
        sup._setup()
        r = sup.runner
        arm = args.arm
        roi = r.box_roi[arm]
        r.move(arm, ["carry", "box_approach"], gripper=r.open_g[arm],
               label=f"{arm} 去盒子", speed=1.0, timeout=5.0)
        r.move(arm, ["box_hover"], gripper=r.open_g[arm],
               label=f"{arm} 盒内观察位", speed=0.85, timeout=4.0)
        frame, det = r.detect_at(arm, roi=roi, force=True)
        print(f"[{arm}] box_hover 检测(ROI): {det}")
        frame2, det2 = r.detect_at(arm, roi=None, force=True)
        print(f"[{arm}] box_hover 检测(全图): {det2}")
        if frame is not None:
            img = r.det.annotate(frame.copy(), det, extra="roi")
            cv2.imwrite(f"/tmp/{args.tag}_{arm}_roi.jpg", img)
        if frame2 is not None:
            img2 = r.det.annotate(frame2.copy(), det2, extra="full")
            cv2.imwrite(f"/tmp/{args.tag}_{arm}_full.jpg", img2)
        print(f"已存 /tmp/{args.tag}_{arm}_roi.jpg / _full.jpg")
        r.move(arm, ["box_approach", "carry", "home"], gripper=r.open_g[arm],
               label=f"{arm} 回 home", speed=1.0, timeout=6.0)
        return 0
    except Exception as exc:
        print(f"失败: {type(exc).__name__}: {exc}")
        try:
            sup.runner.move(args.arm, ["home"], gripper=sup.runner.open_g[args.arm],
                            label=f"{args.arm} 回 home", speed=1.0, timeout=8.0)
        except Exception:
            pass
        return 1
    finally:
        sup._teardown()


if __name__ == "__main__":
    sys.exit(main())
