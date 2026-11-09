#!/usr/bin/env python
"""place_right_into_box.py — 右臂：把桌子上的方块放进盒子。

只跑两段：
  1. 右臂从桌面竖直下探抓方块（盲抓，靠夹爪位置/负载验证）；
  2. 调用主循环的“放入盒子”阶段（box_hover -> box_drop -> 松爪）。

用法（conda activate hupy3.12）：
    python place_right_into_box.py                  # 默认盲抓
    python place_right_into_box.py --extra-depths 0,2,-2
停止：Ctrl+C（会先开爪、再下电）。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

import armcore as C
from dualarm_loop import Supervisor
from pick_right_from_box import head_snapshot


def blind_pick_from_table(runner, arm: str, extra_depths=(0.0, 2.0, -2.0),
                          lift_speed: float = 0.3,
                          grip_hold_s: float = 0.5,
                          pan_off: float = 0.0,
                          exact: bool = True) -> bool:
    """从桌面抓取方块。

    exact=True 时直接下探到手标的 table_pick 位姿（再加 pan_off），
    不再用 hover + 竖直增量（竖直化会把 pan 锁在 hover 值，偏几厘米）。
    """
    r = runner
    r.acquire(arm, "table")
    old_hold = r.cfg["grasp"].get("grip_hold_s", 0.15)
    r.cfg["grasp"]["grip_hold_s"] = float(grip_hold_s)
    try:
        r.log.info("[%s] 阶段：桌面抓取（盲抓，pan 偏移 %+.1f）", arm, pan_off)
        r.move(arm, ["carry", "table_approach"], gripper=r.open_g[arm],
               label=f"{arm} 去桌面", speed=1.0, timeout=5.0)
        r.move(arm, ["table_hover"], gripper=r.open_g[arm],
               label=f"{arm} 桌面观察位", speed=0.9, timeout=4.0)
        for i, extra in enumerate(extra_depths):
            if exact:
                target = r.pose(arm, "table_pick").copy()
                target[0] += float(pan_off) + (float(extra) if i else 0.0)
            else:
                base = r.planned_q(arm, "table_hover").copy()
                d = r.vertical_delta[(arm, "table_hover")].copy()
                d[1] += float(extra)
                target = base + d
                target[0] += float(pan_off)
            target = np.clip(target, -100.0, 100.0)
            r.log.info("[%s] 桌面盲抓第 %d 次：目标 %s", arm, i + 1,
                       C.fmt_q(target))
            r.move(arm, [target], gripper=r.open_g[arm],
                   label=f"{arm} 桌面下探 {i + 1}", precision=True, timeout=6.0)
            if r.close_and_verify(arm):
                r.move(arm, ["table_hover"], gripper=r.close_g[arm],
                       label=f"{arm} 夹起", speed=lift_speed, timeout=8.0)
                if r.lift_verify(arm):
                    r.move(arm, ["carry"], gripper=r.close_g[arm],
                           label=f"{arm} 去搬运位", speed=1.0, timeout=4.0)
                    return True
                r.log.warning("[%s] 桌面盲抓抬起滑脱", arm)
            r.release_gripper(arm)
            if i < len(extra_depths) - 1:
                r.move(arm, ["table_hover"], gripper=r.open_g[arm],
                       label=f"{arm} 回观察位", speed=1.0, timeout=4.0)
        return False
    finally:
        r.cfg["grasp"]["grip_hold_s"] = old_hold
        r.release(arm, "table")


def main() -> int:
    ap = argparse.ArgumentParser(description="右臂把桌面方块放进盒子")
    ap.add_argument("--vision", action="store_true",
                    help="桌面抓取走主循环视觉流程（align_px 对准），默认盲抓")
    ap.add_argument("--pan-off", type=float, default=20.0,
                    help="桌面抓取的 pan 偏移（度）；默认 +20，对应方块当前实际位置")
    ap.add_argument("--extra-depths", default="0,4,-4",
                    help="桌面每次相对配置下探量的额外深度（lift 增大=更深），逗号分隔")
    ap.add_argument("--lift-speed", type=float, default=0.3)
    ap.add_argument("--grip-hold", type=float, default=0.5)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="place_right_box")
    sup = Supervisor(cfg, logger)
    sup._signals()

    ok = False
    try:
        sup._setup()
        runner = sup.runner
        head_snapshot("place_right_before")
        if args.vision:
            runner.phase_pick_from_table("right")
        else:
            depths = tuple(float(v) for v in str(args.extra_depths).split(",") if v.strip())
            if not blind_pick_from_table(runner, "right", extra_depths=depths,
                                         lift_speed=float(args.lift_speed),
                                         grip_hold_s=float(args.grip_hold),
                                         pan_off=float(args.pan_off)):
                raise C.GraspFailed("right 桌面盲抓失败")
        head_snapshot("place_right_picked")
        logger.info("[right] 已从桌面夹起方块")
        runner.phase_place_in_box("right")
        head_snapshot("place_right_after")
        logger.info("完成：方块已放入盒子，右臂已离开")
        ok = True
        return 0
    except C.StopRequested:
        logger.warning("收到停止请求")
        return 130
    except Exception as exc:
        logger.error("失败: %s: %s", type(exc).__name__, exc)
        logger.debug(traceback.format_exc())
        try:
            if sup.runner is not None:
                try:
                    sup.runner.move("right", ["table_hover", "table_approach",
                                              "carry", "home"],
                                    gripper=sup.runner.open_g["right"],
                                    label="right 退出回 home", speed=1.0,
                                    timeout=10.0)
                except Exception:
                    sup.runner.move("right", ["home"],
                                    gripper=sup.runner.open_g["right"],
                                    label="right 直接回 home", speed=1.0,
                                    timeout=10.0)
                head_snapshot("place_right_fail")
        except Exception as exc2:
            logger.warning("失败后回 home 未完成: %s", exc2)
        return 1
    finally:
        sup._teardown()
        logger.info("退出%s", "正常" if ok else "（未完成）")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
