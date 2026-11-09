#!/usr/bin/env python
"""pick_right_from_box.py — 单阶段：右臂把盒子里的方块拿出来。

只执行 dualarm_loop.py 里的“盒内抓出”阶段（carry -> box_approach ->
box_hover -> 视觉对准 -> 分段下探 -> 合爪验证 -> 抬出），不跑完整循环。

抓出后默认把方块放到桌面（右臂 table_hover 点位）再回 home，
这样脚本退出时方块不会悬在半空。也可以用 --place box 放回盒子。

用法（conda activate hupy3.12）：
    python pick_right_from_box.py                 # 取出 -> 放桌面
    python pick_right_from_box.py --place box     # 取出 -> 放回盒子
停止：Ctrl+C（会先开爪、再下电）。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

import armcore as C
from dualarm_loop import Supervisor, save_snapshot


def tracked_pick_from_box(runner, arm: str, extra_depth: float = 5.0,
                          lift_speed: float = 0.2, grip_hold_s: float = 0.4,
                          steps: int = 5) -> bool:
    """腕相机可用时的抓取：分段下探并保持方块在初始像素位置。

    与主循环的 phase_pick_from_box 区别：不做 align_at_hover 的
    align_px 预对准（当前 align_px 与实际画面偏差大，会把爪子带偏），
    只跟踪 hover 那一刻方块的像素位置，边走边对中。
    extra_depth 是相对配置下探量的额外深度（lift 增大 = 更深）。
    """
    r = runner
    r.acquire(arm, "box")
    old_hold = r.cfg["grasp"].get("grip_hold_s", 0.15)
    r.cfg["grasp"]["grip_hold_s"] = float(grip_hold_s)
    try:
        r.log.info("[%s] 阶段：盒内抓出（跟踪下探，额外深度 %+.1f）", arm, extra_depth)
        r.move(arm, ["carry", "box_approach"], gripper=r.open_g[arm],
               label=f"{arm} 去盒子上方", speed=1.0, timeout=5.0)
        r.move(arm, ["box_hover"], gripper=r.open_g[arm],
               label=f"{arm} 盒内观察位", speed=0.85, timeout=4.0)
        delta = r.vertical_delta[(arm, "box_hover")].copy()
        delta[1] += float(extra_depth)
        r.descend_tracking(arm, "box_hover", delta, gripper=r.open_g[arm],
                           roi=r.box_roi[arm], steps=steps,
                           label=f"{arm} 盒内下探")
        frame, det = r.detect_at(arm, roi=r.box_roi[arm], force=True)
        r.log.info("[%s] 合爪前像素: %s", arm, det)
        if frame is not None:
            save_snapshot(r.cfg, r.det, frame, det,
                          f"{arm}_before_close", r.log)
        if r.close_and_verify(arm):
            r.move(arm, ["box_hover"], gripper=r.close_g[arm],
                   label=f"{arm} 夹出", speed=lift_speed, timeout=8.0)
            if r.lift_verify(arm):
                r.move(arm, ["box_approach", "carry"], gripper=r.close_g[arm],
                       label=f"{arm} 取出到搬运位", speed=1.0, timeout=5.0)
                return True
            r.log.warning("[%s] 跟踪抓取抬起滑脱", arm)
        r.release_gripper(arm)
        return False
    finally:
        r.cfg["grasp"]["grip_hold_s"] = old_hold
        r.release(arm, "box")


def blind_pick_from_box(runner, arm: str, extra_depths=(0.0, 3.0),
                        lift_speed: float = 0.45,
                        grip_hold_s: float = 0.15,
                        roll_deg: float = 0.0) -> bool:
    """不用腕相机，按标定位姿竖直下探抓取（腕相机掉线时的兜底）。

    extra_depths 是相对当前 vertical_delta 的额外下探量（lift 增大 = 更低）。
    第一次就是配置里现在的下探高度；失败后再往深/浅各试一次。
    """
    r = runner
    r.acquire(arm, "box")
    try:
        r.log.info("[%s] 阶段：盒内抓出（盲抓，不用腕相机）", arm)
        r.move(arm, ["carry", "box_approach"], gripper=r.open_g[arm],
               label=f"{arm} 去盒子上方", speed=1.0, timeout=5.0)
        r.move(arm, ["box_hover"], gripper=r.open_g[arm],
               label=f"{arm} 盒内观察位", speed=0.85, timeout=4.0)
        if abs(roll_deg) > 1e-6:
            q_hover = r.pose(arm, "box_hover").copy()
            q_hover[4] += float(roll_deg)
            r.move(arm, [np.clip(q_hover, -100.0, 100.0)], gripper=r.open_g[arm],
                   label=f"{arm} hover 转腕 {roll_deg:+.0f}", speed=0.8, timeout=4.0)
        base = r.planned_q(arm, "box_hover").copy()
        delta = r.vertical_delta[(arm, "box_hover")].copy()
        old_hold = r.cfg["grasp"].get("grip_hold_s", 0.15)
        r.cfg["grasp"]["grip_hold_s"] = float(grip_hold_s)
        for i, extra in enumerate(extra_depths):
            if i:
                r.move(arm, ["box_hover"], gripper=r.open_g[arm],
                       label=f"{arm} 回观察位", speed=1.0, timeout=4.0)
            d = delta.copy()
            d[1] += float(extra)
            target = np.clip(base + d, -100.0, 100.0)
            r.log.info("[%s] 盲抓第 %d 次：额外下探 %+.1f 抬升速度 %.2f",
                       arm, i + 1, extra, lift_speed)
            r.move(arm, [target], gripper=r.open_g[arm],
                   label=f"{arm} 盲抓下探 {i + 1}", precision=True, timeout=6.0)
            if r.close_and_verify(arm):
                r.move(arm, ["box_hover"], gripper=r.close_g[arm],
                       label=f"{arm} 夹出", speed=lift_speed, timeout=8.0)
                if r.lift_verify(arm):
                    r.move(arm, ["box_approach", "carry"], gripper=r.close_g[arm],
                           label=f"{arm} 取出到搬运位", speed=1.0, timeout=5.0)
                    r.cfg["grasp"]["grip_hold_s"] = old_hold
                    return True
                r.log.warning("[%s] 盲抓抬起滑脱", arm)
            r.release_gripper(arm)
        r.cfg["grasp"]["grip_hold_s"] = old_hold
        return False
    finally:
        r.release(arm, "box")


def head_snapshot(tag: str) -> None:
    """用头部相机（/dev/video4）拍一张，便于人工核对；没有该相机就跳过。"""
    try:
        import cv2
        cap = cv2.VideoCapture("/dev/video4")
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                cv2.imwrite(f"/tmp/{tag}.jpg", frame)
        cap.release()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="右臂从盒子里取方块（单阶段）")
    ap.add_argument("--place", choices=["table", "box"], default="table",
                    help="取出后放哪里：table=桌面（默认），box=放回盒子")
    ap.add_argument("--blind", action="store_true",
                    help="腕相机不可用时按标定位姿盲抓")
    ap.add_argument("--track", action="store_true",
                    help="腕相机可用时跟踪下探（保持方块初始像素，不做 align_px 预对准）")
    ap.add_argument("--extra-depth", type=float, default=5.0,
                    help="--track 模式相对配置下探量的额外深度（lift 增大=更深）")
    ap.add_argument("--steps", type=int, default=5,
                    help="--track 模式下探分段数")
    ap.add_argument("--extra-depths", default="0,3",
                    help="盲抓每次相对配置下探量的额外深度（lift 增大=更深），逗号分隔")
    ap.add_argument("--lift-speed", type=float, default=None,
                    help="盲抓抬升速度倍率（默认取 config grasp.lift_speed）")
    ap.add_argument("--grip-hold", type=float, default=0.15,
                    help="合爪后停留时间（秒），让方块落稳再抬")
    ap.add_argument("--roll", type=float, default=0.0,
                    help="盲抓前先转腕的角度（度，正负见探测图）")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="pick_right_box")
    sup = Supervisor(cfg, logger)
    sup._signals()

    ok = False
    try:
        sup._setup()
        runner = sup.runner
        if args.blind:
            head_snapshot("pick_right_before")
            depths = tuple(float(v) for v in str(args.extra_depths).split(",") if v.strip())
            lift_speed = (float(args.lift_speed) if args.lift_speed is not None
                          else float(cfg["grasp"].get("lift_speed", 0.45)))
            if not blind_pick_from_box(runner, "right", extra_depths=depths,
                                       lift_speed=lift_speed,
                                       grip_hold_s=float(args.grip_hold),
                                       roll_deg=float(args.roll)):
                raise C.GraspFailed("right 盲抓多次都失败")
        elif args.track:
            head_snapshot("pick_right_before")
            lift_speed = (float(args.lift_speed) if args.lift_speed is not None
                          else 0.2)
            if not tracked_pick_from_box(runner, "right",
                                         extra_depth=float(args.extra_depth),
                                         lift_speed=lift_speed,
                                         grip_hold_s=float(args.grip_hold),
                                         steps=int(args.steps)):
                raise C.GraspFailed("right 跟踪抓取失败")
        else:
            runner.phase_pick_from_box("right", attempt_note="（手动测试）")
        head_snapshot("pick_right_after")
        logger.info("[right] 已把方块从盒子里夹出并抬到搬运位")
        if args.place == "table":
            runner.phase_place_on_table("right")
            logger.info("完成：方块已放到桌面（右臂 table_hover 点位），爪子已松开")
        else:
            runner.phase_place_in_box_again("right")
            logger.info("完成：方块已放回盒子，爪子已松开")
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
                    sup.runner.move("right", ["box_hover", "box_approach", "carry", "home"],
                                    gripper=sup.runner.open_g["right"],
                                    label="right 沿盒口退出回 home", speed=1.0,
                                    timeout=8.0)
                except Exception:
                    sup.runner.move("right", ["home"], gripper=sup.runner.open_g["right"],
                                    label="right 直接回 home", speed=1.0, timeout=8.0)
                head_snapshot("pick_right_fail")
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
