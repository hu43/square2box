#!/usr/bin/env python
"""pick_right_from_table.py — 单阶段：右臂把桌面上的方块夹起来。

流程：carry -> table_approach -> table_hover（视觉对准）-> 竖直下探
-> 合爪验证 -> 夹起 -> carry，然后放回桌面同一个点并回 home。

用法（conda activate hupy3.12）：
    python pick_right_from_table.py
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


def blind_pick_from_table(runner, arm: str, lift_speed: float = 0.3,
                          grip_hold_s: float = 0.4) -> bool:
    """腕相机不可用时，按标定位姿竖直下探抓桌面方块。

    方块应位于标定的 table_pick 点（例如右臂刚放下的位置）。
    """
    r = runner
    r.acquire(arm, "table")
    old_hold = r.cfg["grasp"].get("grip_hold_s", 0.15)
    r.cfg["grasp"]["grip_hold_s"] = float(grip_hold_s)
    try:
        r.log.info("[%s] 阶段：桌面抓取（盲抓，不用腕相机）", arm)
        r.move(arm, ["carry", "table_approach"], gripper=r.open_g[arm],
               label=f"{arm} 去桌面", speed=1.0)
        r.move(arm, ["table_hover"], gripper=r.open_g[arm],
               label=f"{arm} 桌面观察位", speed=0.9, timeout=4.0)
        base = r.planned_q(arm, "table_hover").copy()
        delta = r.vertical_delta[(arm, "table_hover")].copy()
        deepen = float(r.cfg["grasp"].get("retry_deepen", 0.8))
        for attempt in range(0, 2):
            if attempt:
                r.move(arm, ["table_hover"], gripper=r.open_g[arm],
                       label=f"{arm} 回观察位", speed=1.0, timeout=4.0)
            d = delta.copy()
            d[1] += deepen * attempt
            target = np.clip(base + d, -100.0, 100.0)
            r.log.info("[%s] 桌面盲抓第 %d 次（额外下探 %.1f）", arm, attempt + 1,
                       deepen * attempt)
            r.move(arm, [target], gripper=r.open_g[arm],
                   label=f"{arm} 桌面下探 {attempt + 1}", precision=True, timeout=6.0)
            if r.close_and_verify(arm):
                r.move(arm, ["table_hover"], gripper=r.close_g[arm],
                       label=f"{arm} 夹起", speed=lift_speed, timeout=8.0)
                if r.lift_verify(arm):
                    r.move(arm, ["carry"], gripper=r.close_g[arm],
                           label=f"{arm} 去搬运位", speed=1.0, timeout=4.0)
                    return True
                r.log.warning("[%s] 桌面盲抓抬起滑脱", arm)
            r.release_gripper(arm)
        return False
    finally:
        r.cfg["grasp"]["grip_hold_s"] = old_hold
        r.release(arm, "table")


def main() -> int:
    ap = argparse.ArgumentParser(description="右臂桌面抓取（单阶段）")
    ap.add_argument("--blind", action="store_true",
                    help="腕相机不可用时按标定位姿盲抓")
    ap.add_argument("--lift-speed", type=float, default=0.3)
    ap.add_argument("--grip-hold", type=float, default=0.4)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="pick_right_table")
    sup = Supervisor(cfg, logger)
    sup._signals()

    ok = False
    try:
        sup._setup()
        runner = sup.runner
        head_snapshot("pick_right_table_before")
        if args.blind:
            if not blind_pick_from_table(runner, "right",
                                         lift_speed=float(args.lift_speed),
                                         grip_hold_s=float(args.grip_hold)):
                raise C.GraspFailed("right 桌面盲抓失败")
        else:
            runner.phase_pick_from_table("right")
        head_snapshot("pick_right_table_after")
        logger.info("[right] 已把桌面上的方块夹起并抬到搬运位")
        runner.phase_place_on_table("right")
        head_snapshot("pick_right_table_done")
        logger.info("完成：方块已放回桌面，爪子已松开")
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
                    sup.runner.move("right", ["home"], gripper=sup.runner.open_g["right"],
                                    label="right 直接回 home", speed=1.0, timeout=10.0)
                head_snapshot("pick_right_table_fail")
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
