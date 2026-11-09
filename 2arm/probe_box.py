#!/usr/bin/env python
"""probe_box.py — 下探到盒内指定深度，拍腕相机图（不合爪），再退回 home。

用于按图像微调右臂盒内抓取姿态。目标 = box_hover + 竖直增量
（已含 config 的 box_pick_height_offset）+ 额外偏移。

用法：
  python probe_box.py --extra-depth 3              # 回到手标 box_pick 原始深度
  python probe_box.py --extra-depth 3 --pan 2 --lift -1
图片存 /tmp/probe_<tag>.jpg，终端打印检测结果。
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
    ap.add_argument("--scene", choices=["box", "table"], default="box",
                    help="box=盒内下探，table=桌面下探")
    ap.add_argument("--extra-depth", type=float, default=3.0,
                    help="相对配置竖直增量的额外深度（lift 增大=更深）")
    ap.add_argument("--pan", type=float, default=0.0)
    ap.add_argument("--lift", type=float, default=0.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="probe_box")
    sup = Supervisor(cfg, logger)
    sup._signals()
    arm = args.arm
    box = args.scene == "box"
    hover_key = "box_hover" if box else "table_hover"
    approach = ["carry", "box_approach"] if box else ["carry", "table_approach"]
    retreat = ["box_hover", "box_approach", "carry", "home"] if box else \
              ["table_hover", "table_approach", "carry", "home"]
    roi = r_box_roi = None if box else None  # 占位，setup 后取
    try:
        sup._setup()
        r = sup.runner
        roi = r.box_roi[arm] if box else None
        r.move(arm, approach, gripper=r.open_g[arm],
               label=f"{arm} 去{'盒子' if box else '桌面'}上方", speed=1.0, timeout=5.0)
        r.move(arm, [hover_key], gripper=r.open_g[arm],
               label=f"{arm} 观察位", speed=0.85, timeout=4.0)
        frame0, det0 = r.detect_at(arm, roi=roi, force=True)
        r.log.info("[%s] hover 检测: %s", arm, det0)
        if frame0 is not None:
            cv2.imwrite(f"/tmp/{args.tag}_hover.jpg",
                        r.det.annotate(frame0.copy(), det0))
        d = r.vertical_delta[(arm, hover_key)].copy()
        d[1] += float(args.extra_depth)
        d[0] += float(args.pan)
        d[1] += float(args.lift)
        if abs(args.roll) > 1e-6:
            # 先在 hover 上空转腕，再带这个 roll 下探，避免在方块旁边旋转
            q_hover = r.pose(arm, hover_key).copy()
            q_hover[4] += float(args.roll)
            r.move(arm, [np.clip(q_hover, -100, 100)], gripper=r.open_g[arm],
                   label=f"{arm} hover 转腕 {args.roll:+.0f}", speed=0.8, timeout=4.0)
        target = np.clip(r.planned_q(arm, hover_key) + d, -100, 100)
        r.log.info("[%s] 下探目标: %s", arm, C.fmt_q(target))
        r.move(arm, [target], gripper=r.open_g[arm],
               label=f"{arm} 探测下探", precision=True, timeout=8.0)
        frame, det = r.detect_at(arm, roi=roi, force=True)
        r.log.info("[%s] 深处检测: %s", arm, det)
        if frame is not None:
            cv2.imwrite(f"/tmp/{args.tag}.jpg",
                        r.det.annotate(frame.copy(), det,
                                       extra=f"pan{args.pan:+.0f} lift{args.lift:+.0f} "
                                             f"depth{args.extra_depth:+.0f}"))
            cv2.imwrite(f"/tmp/{args.tag}_raw.jpg", frame)
        capture_head(f"/tmp/{args.tag}_head.jpg")
        # 停机前先把爪张开、退回安全位
        r.move(arm, retreat, gripper=r.open_g[arm],
               label=f"{arm} 回 home", speed=1.0, timeout=8.0)
        return 0
    except Exception as exc:
        print(f"失败: {type(exc).__name__}: {exc}")
        try:
            sup.runner.move(arm, retreat, gripper=sup.runner.open_g[arm],
                            label=f"{arm} 回 home", speed=1.0, timeout=10.0)
        except Exception:
            pass
        return 1
    finally:
        sup._teardown()


if __name__ == "__main__":
    sys.exit(main())
