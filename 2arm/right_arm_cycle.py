#!/usr/bin/env python
"""right_arm_cycle.py — 右臂完整循环：盒内取出 -> 放桌面 -> 桌面抓取 -> 放回盒内。

合并自两个单阶段工具：
  pick_right_from_box.py   （右臂把盒子里的方块拿出来，放到桌面）
  place_right_into_box.py  （右臂把桌子上的方块放进盒子）

流程（全部只用右臂）：
  1. 盒内取出：box_hover -> 竖直下探 -> 合爪验证 -> 慢速抬出；
  2. 放到桌面：carry -> table_hover -> 竖直放下 -> 松爪 -> 回 home；
  3. 桌面抓取：按 table_pick 位姿（可带 pan 偏移）下探 -> 合爪验证 -> 夹起；
  4. 放回盒内：box_hover -> box_drop -> 松爪 -> 离开。

针对方块摆放偏差做了回退：
  * 盒内取出依次尝试腕转 0° / +15° / -15°（方块朝向不同时对齐爪口）；
  * 桌面抓取依次尝试 pan 偏移 0 / +20 / -16（方块被挪动过时也能抓到）。

用法（conda activate hupy3.12）：
    python right_arm_cycle.py                 # 完整跑一遍
    python right_arm_cycle.py --box-rolls 15  # 只试腕转 15°
    python right_arm_cycle.py --table-pans 0  # 只按标定位姿抓桌面
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
from pick_right_from_box import blind_pick_from_box, head_snapshot
from place_right_into_box import blind_pick_from_table


def floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in str(text).split(",") if v.strip())


# 手标抓取图（calib/right_table_pick.jpg）里方块的像素位置：方块出现在
# 这里时，两指正好夹在方块两侧。
TABLE_TARGET_PX = (366.5, 322.5)
# 在 table_pick 位姿附近实测的像素响应（px/度）：
#   pan +1   -> (-3.46, +2.94)
#   elbow +1 -> (+3.12, -10.18)   （elbow 减小 = 手往前伸、y 增大）
_TABLE_J_INV = ((-0.3908, -0.1198), (-0.1129, -0.1328))


def servo_pick_table(runner, arm: str, lift_speed: float = 0.3,
                     grip_hold_s: float = 0.5, max_iters: int = 5) -> bool:
    """用腕相机把方块伺服到标定抓取像素，再合爪抓取。

    只调 pan（左右）和 elbow（前后/上下），增益是现场实测的局部响应。
    """
    r = runner
    r.acquire(arm, "table")
    old_hold = r.cfg["grasp"].get("grip_hold_s", 0.15)
    r.cfg["grasp"]["grip_hold_s"] = float(grip_hold_s)
    try:
        r.log.info("[%s] 阶段：桌面抓取（视觉伺服）", arm)
        r.move(arm, ["carry", "table_approach"], gripper=r.open_g[arm],
               label=f"{arm} 去桌面", speed=1.0, timeout=5.0)
        r.move(arm, ["table_hover"], gripper=r.open_g[arm],
               label=f"{arm} 桌面观察位", speed=0.9, timeout=4.0)
        q = r.pose(arm, "table_pick").copy()
        det = None
        for it in range(max_iters):
            r.move(arm, [np.clip(q, -100, 100)], gripper=r.open_g[arm],
                   label=f"{arm} 伺服下探 {it + 1}", precision=True, timeout=6.0)
            frame, det = r.detect_at(arm, force=True)
            if det is None:
                r.log.warning("[%s] 伺服第 %d 次没看到方块", arm, it + 1)
                break
            ex = TABLE_TARGET_PX[0] - det.cx
            ey = TABLE_TARGET_PX[1] - det.cy
            r.log.info("[%s] 伺服第 %d 次: c=(%.0f,%.0f) area=%.0f 误差=(%+.0f,%+.0f)",
                       arm, it + 1, det.cx, det.cy, det.area, ex, ey)
            if frame is not None:
                save_snapshot(r.cfg, r.det, frame, det,
                              f"{arm}_table_servo_{it + 1}", r.log)
            if abs(ex) <= 20.0 and abs(ey) <= 30.0:
                break
            dp = _TABLE_J_INV[0][0] * ex + _TABLE_J_INV[0][1] * ey
            de = _TABLE_J_INV[1][0] * ex + _TABLE_J_INV[1][1] * ey
            q[0] = float(np.clip(q[0] + np.clip(0.8 * dp, -10.0, 10.0), -100.0, 100.0))
            q[2] = float(np.clip(q[2] + np.clip(0.8 * de, -6.0, 6.0), -100.0, 100.0))
        frame, det = r.detect_at(arm, force=True)
        r.log.info("[%s] 合爪前: %s", arm, det)
        if frame is not None:
            save_snapshot(r.cfg, r.det, frame, det, f"{arm}_table_before_close", r.log)
        if r.close_and_verify(arm):
            r.move(arm, ["table_hover"], gripper=r.close_g[arm],
                   label=f"{arm} 夹起", speed=lift_speed, timeout=8.0)
            if r.lift_verify(arm):
                r.move(arm, ["carry"], gripper=r.close_g[arm],
                       label=f"{arm} 去搬运位", speed=1.0, timeout=4.0)
                return True
            r.log.warning("[%s] 桌面伺服抓取抬起滑脱", arm)
        r.release_gripper(arm)
        return False
    finally:
        r.cfg["grasp"]["grip_hold_s"] = old_hold
        r.release(arm, "table")


def main() -> int:
    ap = argparse.ArgumentParser(description="右臂完整循环：盒内取出 -> 桌面 -> 放回盒内")
    ap.add_argument("--box-depth", type=float, default=3.0,
                    help="盒内下探相对配置竖直增量的额外深度（抵消 box_pick_height_offset）")
    ap.add_argument("--box-rolls", default="0,15,-15",
                    help="盒内抓取依次尝试的腕转角度，逗号分隔")
    ap.add_argument("--table-pans", default="20,-16",
                    help="桌面抓取回退时依次尝试的 pan 偏移（方块被外力挪动时用）")
    ap.add_argument("--skip-out", action="store_true",
                    help="跳过第 1~2 步（假定方块已在桌面），只跑桌面->盒子")
    ap.add_argument("--lift-speed", type=float, default=0.3,
                    help="抓取后抬升速度倍率")
    ap.add_argument("--grip-hold", type=float, default=0.5,
                    help="合爪后停留时间（秒）")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="right_arm_cycle")
    sup = Supervisor(cfg, logger)
    sup._signals()
    arm = "right"

    ok = False
    try:
        sup._setup()
        r = sup.runner
        head_snapshot("cycle_before")

        # ---- 1. 盒内取出 ----
        if args.skip_out:
            logger.info("跳过第 1~2 步（假定方块已在桌面）")
        else:
            logger.info("=" * 60)
            logger.info("第 1 步：右臂从盒子里取出方块")
            picked = False
            for roll in floats(args.box_rolls):
                logger.info("[right] 盒内抓取：腕转 %+.0f°", roll)
                if blind_pick_from_box(r, arm, extra_depths=(float(args.box_depth),),
                                       lift_speed=float(args.lift_speed),
                                       grip_hold_s=float(args.grip_hold),
                                       roll_deg=roll):
                    picked = True
                    break
                logger.warning("[right] 腕转 %+.0f° 没抓住，换下一个角度", roll)
                r.move(arm, ["box_hover", "box_approach", "carry"],
                       gripper=r.open_g[arm], label="right 退出盒子",
                       speed=1.0, timeout=6.0)
            if not picked:
                raise C.GraspFailed("right 盒内抓取所有腕转角度都失败")
            logger.info("[right] 已把方块从盒子里取出")
            head_snapshot("cycle_out_of_box")

            # ---- 2. 放到桌面 ----
            logger.info("=" * 60)
            logger.info("第 2 步：右臂把方块放到桌面")
            r.phase_place_on_table(arm)
            logger.info("[right] 方块已放到桌面")
            head_snapshot("cycle_on_table")

        # ---- 3. 桌面抓取 ----
        logger.info("=" * 60)
        logger.info("第 3 步：右臂从桌面抓取方块")
        # 1) 视觉伺服：把方块对到标定抓取像素；
        picked = servo_pick_table(r, arm, lift_speed=float(args.lift_speed),
                                  grip_hold_s=float(args.grip_hold))
        # 2) 回退：沿“放桌面”的同一路径反向下探（方块正好落在释放位姿时）；
        if not picked:
            r.move(arm, ["table_hover", "table_approach", "carry"],
                   gripper=r.open_g[arm], label="right 离开桌面",
                   speed=1.0, timeout=6.0)
            picked = blind_pick_from_table(r, arm, extra_depths=(0.0,),
                                           lift_speed=float(args.lift_speed),
                                           grip_hold_s=float(args.grip_hold),
                                           pan_off=0.0, exact=False)
        # 3) 再回退：按手标 table_pick + pan 偏移扫描。
        if not picked:
            r.move(arm, ["table_hover", "table_approach", "carry"],
                   gripper=r.open_g[arm], label="right 离开桌面",
                   speed=1.0, timeout=6.0)
            for pan in floats(args.table_pans):
                logger.info("[right] 桌面抓取回退：pan 偏移 %+.0f°", pan)
                if blind_pick_from_table(r, arm, extra_depths=(0.0,),
                                         lift_speed=float(args.lift_speed),
                                         grip_hold_s=float(args.grip_hold),
                                         pan_off=pan, exact=True):
                    picked = True
                    break
                logger.warning("[right] pan %+.0f° 没抓住", pan)
                r.move(arm, ["table_hover", "table_approach", "carry"],
                       gripper=r.open_g[arm], label="right 离开桌面",
                       speed=1.0, timeout=6.0)
        if not picked:
            raise C.GraspFailed("right 桌面抓取失败（伺服 + 释放位姿 + pan 扫描）")
        logger.info("[right] 已从桌面夹起方块")
        head_snapshot("cycle_picked_from_table")

        # ---- 4. 放回盒内 ----
        logger.info("=" * 60)
        logger.info("第 4 步：右臂把方块放回盒子")
        r.phase_place_in_box(arm)
        logger.info("[right] 方块已放回盒子")
        r.move(arm, ["home"], gripper=r.open_g[arm],
               label="right 回 home", speed=1.0, timeout=6.0)
        head_snapshot("cycle_done")
        logger.info("完整循环完成：盒内 -> 桌面 -> 盒内")
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
                sup.runner.move(arm, ["carry", "home"], gripper=sup.runner.open_g[arm],
                                label="right 退出回 home", speed=1.0, timeout=10.0)
                head_snapshot("cycle_fail")
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
