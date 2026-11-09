#!/usr/bin/env python
"""four_step_cycle.py — 双臂四步循环（左臂放盒 / 右臂取出放桌 / 右臂从桌放回 / 左臂取出）。

参考 right_arm_cycle.py（右臂单臂循环）与 square2box_loop.py（单臂
pick -> place_in_box -> pick_from_box -> place_on_table）的流程，
复用 dualarm_loop.LoopRunner 里已经调好的阶段：

  1. 左臂：桌面夹起绿色方块 -> 放入盒子      （phase_place_in_box）
  2. 右臂：从盒子里夹出方块 -> 放到桌面      （phase_place_on_table）
  3. 右臂：从桌面抓取方块 -> 放回盒中        （servo_pick_table + phase_place_in_box）
  4. 左臂：把盒子里的方块夹出来               （phase_pick_from_box）

第 1 步的桌面抓取不用主循环的视觉对准：左臂 align_px 是在“手持方块”
的画面里标出来的，对真实的 table_hover 偏差 270~320px，视觉对准会
把臂带偏。实测可靠的是按手标 table_pick 位姿竖直下探（logs/left_sweep.log
与 logs/leftpan.log 都验证过），所以这里按 pan 偏移扫描，最后才退回
主循环的视觉流程。

第 3 步的桌面抓取同理不能套用左臂参数：right_arm_cycle.TABLE_TARGET_PX
与 _TABLE_J_INV 是右臂在 table_pick 位姿实测的像素响应，这里直接复用这
两个常量，但不用 servo_pick_table——它的退出条件是 |ex|<=20 且 |ey|<=30，
实测 y 残差 22px 就提前合爪，两指落在方块下方夹空。servo_table_strict
用同一组增益、把阈值收到 ±12px 并多给几次迭代。伺服失败再退回“沿放桌面
路径反向下探”、手标 table_pick 的 pan 扫描，最后才是主循环视觉流程。

盒内抓取：日志显示视觉对准在盒里经常夹空（腕相机还时不时掉线），
盲抓（标定 pan + 腕转 0/±15°）才是稳定路径，所以盲抓优先、视觉兜底。

收尾：第 4 步完成后方块在左爪里。若直接退出，Supervisor 的停机保护会
把“左臂夹着方块”当成 in_box 状态又放回盒子里，等于把第 4 步做回去了。
所以默认再走一次 phase_place_on_table，把方块放到盒子旁边的桌面上，
程序结束时两臂都不夹东西、方块明确在盒外。

用法（conda activate hupy3.12）：
    python four_step_cycle.py                  # 完整四步
    python four_step_cycle.py --start-box      # 方块已在盒内，只跑 2~4 步
    python four_step_cycle.py --stop-after 2   # 只跑到第 2 步（调试）
    python four_step_cycle.py --keep-in-hand    # 最后不放到桌面（方块留在左爪）
停止：Ctrl+C（会先开爪、再下电）。
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import armcore as C
from dualarm_loop import Supervisor, save_snapshot
from pick_right_from_box import blind_pick_from_box, head_snapshot
from place_right_into_box import blind_pick_from_table
from right_arm_cycle import _TABLE_J_INV, TABLE_TARGET_PX, floats

# 上一步刚刚"放下并保持"的位姿：下一步原地合爪就能重新夹住，方块不会被碰走。
_REGRASP_READY: dict[str, bool] = {}
_REGRASP_POSE: dict[str, "np.ndarray"] = {}


def _raise_clear(r, arm: str, label: str = "抬离") -> None:
    """先在原地竖直抬离桌面/盒底，再横移。

    实测：张开的夹爪从 table_pick 高度直接走 carry 会把桌面方块扫走
    （上一次失败收尾时方块就是这样被推进盒子的）。竖直增量的方向是
    shoulder_lift 增大=下降、elbow_flex 减小=下降，所以抬离要反过来。
    """
    a = r.arms[arm]
    try:
        q = np.asarray(a.read_joints(), dtype=float).copy()
    except C.StopRequested:
        raise
    except Exception as exc:
        r.log.warning("[%s] 读关节失败，跳过%s: %s", arm, label, exc)
        return
    if q.size < 5:
        return
    q[1] = float(np.clip(q[1] - 35.0, -100.0, 100.0))
    q[2] = float(np.clip(q[2] + 30.0, -100.0, 100.0))
    try:
        r.move(arm, [q], gripper=r.close_g[arm], label=f"{arm} {label}",
               speed=1.0, timeout=8.0)
    except C.StopRequested:
        raise
    except Exception as exc:
        r.log.warning("[%s] %s未完成: %s", arm, label, exc)


def _retreat(r, arm: str, keys, label: str, raise_first: bool = True):
    if raise_first:
        _raise_clear(r, arm, label="抬离")
    try:
        r.move(arm, list(keys), gripper=r.open_g[arm], label=label,
               speed=1.0, timeout=6.0)
    except C.StopRequested:
        raise
    except Exception as exc:
        r.log.warning("[%s] %s 未完成: %s", arm, label, exc)


def servo_table_strict(r, arm: str, lift_speed: float, grip_hold_s: float,
                       tol_px: float = 12.0, max_iters: int = 8) -> bool:
    """右臂桌面视觉伺服抓取，收敛比 servo_pick_table 更严。

    right_arm_cycle.servo_pick_table 在 |ex|<=20 且 |ey|<=30 时就合爪，实测
    y 残差 22px 时两指落在方块下方夹空（pos 等于空夹基线）。这里沿用同一
    组标定增益（TABLE_TARGET_PX / _TABLE_J_INV 是右臂 table_pick 位姿的实测
    响应），但把收敛阈值收到 tol_px，并允许更多次迭代。
    """
    r.acquire(arm, "table")
    old_hold = r.cfg["grasp"].get("grip_hold_s", 0.15)
    r.cfg["grasp"]["grip_hold_s"] = float(grip_hold_s)
    try:
        r.log.info("[%s] 阶段：桌面抓取（严格伺服，容差 %.0fpx）", arm, tol_px)
        r.move(arm, ["carry", "table_approach"], gripper=r.open_g[arm],
               label=f"{arm} 去桌面", speed=1.0, timeout=5.0)
        r.move(arm, ["table_hover"], gripper=r.open_g[arm],
               label=f"{arm} 桌面观察位", speed=0.9, timeout=4.0)
        q = r.pose(arm, "table_pick").copy()
        converged = False
        for it in range(max_iters):
            r.move(arm, [np.clip(q, -100, 100)], gripper=r.open_g[arm],
                   label=f"{arm} 伺服下探 {it + 1}", precision=True, timeout=6.0)
            frame, det = r.detect_at(arm, force=True)
            if det is None:
                r.log.warning("[%s] 严格伺服第 %d 次没看到方块", arm, it + 1)
                break
            ex = TABLE_TARGET_PX[0] - det.cx
            ey = TABLE_TARGET_PX[1] - det.cy
            r.log.info("[%s] 严格伺服第 %d 次: c=(%.0f,%.0f) area=%.0f "
                       "误差=(%+.0f,%+.0f)", arm, it + 1, det.cx, det.cy,
                       det.area, ex, ey)
            if frame is not None:
                save_snapshot(r.cfg, r.det, frame, det,
                              f"{arm}_table_strict_{it + 1}", r.log)
            if abs(ex) <= tol_px and abs(ey) <= tol_px:
                converged = True
                break
            dp = _TABLE_J_INV[0][0] * ex + _TABLE_J_INV[0][1] * ey
            de = _TABLE_J_INV[1][0] * ex + _TABLE_J_INV[1][1] * ey
            q[0] = float(np.clip(q[0] + np.clip(0.8 * dp, -10.0, 10.0),
                                 -100.0, 100.0))
            q[2] = float(np.clip(q[2] + np.clip(0.8 * de, -6.0, 6.0),
                                 -100.0, 100.0))
        if not converged:
            r.log.warning("[%s] 严格伺服没收敛到 %.0fpx，仍按当前位姿合爪", arm,
                          tol_px)
        if r.close_and_verify(arm):
            r.move(arm, ["table_hover"], gripper=r.close_g[arm],
                   label=f"{arm} 夹起", speed=lift_speed, timeout=8.0)
            if r.lift_verify(arm):
                r.move(arm, ["carry"], gripper=r.close_g[arm],
                       label=f"{arm} 去搬运位", speed=1.0, timeout=4.0)
                return True
            r.log.warning("[%s] 严格伺服抬起滑脱", arm)
        r.release_gripper(arm)
        return False
    finally:
        r.cfg["grasp"]["grip_hold_s"] = old_hold
        r.release(arm, "table")


def _regrasp_after_place(r, arm: str, lift_speed: float,
                         grip_hold: float) -> bool:
    """上一步刚把方块放下并保持原位：原地合爪重新夹住。

    比"退回观察位再重新下探"稳得多——重新下探实测常差几毫米夹空，而
    原地合爪时两指本来就还在方块两侧。
    """
    if not _REGRASP_READY.pop(arm, False):
        return False
    base = _REGRASP_POSE.pop(arm, None)
    r.log.info("[%s] 桌面抓取：原地合爪（上一步刚放下）", arm)
    old_hold = r.cfg["grasp"].get("grip_hold_s", 0.15)
    r.cfg["grasp"]["grip_hold_s"] = float(grip_hold)
    r.acquire(arm, "table")
    try:
        if not r.close_and_verify(arm):
            r.release_gripper(arm)
            return False
        lift_to = base if base is not None else r.pose(arm, "table_hover")
        r.move(arm, [np.clip(lift_to, -100.0, 100.0)], gripper=r.close_g[arm],
               label=f"{arm} 原地夹起", speed=lift_speed, timeout=8.0)
        if r.lift_verify(arm):
            r.move(arm, ["carry"], gripper=r.close_g[arm],
                   label=f"{arm} 去搬运位", speed=1.0, timeout=4.0)
            return True
        r.log.warning("[%s] 原地合爪抬起滑脱", arm)
        r.release_gripper(arm)
        return False
    finally:
        r.cfg["grasp"]["grip_hold_s"] = old_hold
        r.release(arm, "table")


def pick_from_table_right(r, pans, lift_speed: float, grip_hold: float,
                          tol_px: float = 12.0) -> None:
    """右臂从桌面抓起方块：原地合爪 -> 标定位姿 -> pan 扫描 -> 视觉兜底。

    第 2 步放下后会保持原位（hold_for_regrasp），所以这里先原地合爪重新
    夹住——不重新下探，就没有"差几毫米夹空"的问题。
    视觉伺服（servo_table_strict）用的是重标前的像素目标/雅可比，会把臂
    带到够不到的位姿（precision 动作超时抛 PhaseError），只作最后兜底
    并且不让它中断流程。
    """
    try:
        if _regrasp_after_place(r, "right", lift_speed, grip_hold):
            return
    except C.StopRequested:
        raise
    except Exception as exc:
        r.log.warning("[right] 原地合爪异常(%s: %s)，继续标定路径",
                      type(exc).__name__, exc)
    r.log.info("[right] 桌面抓取：按 table_pick 位姿竖直下探")
    if _table_pick_vertical(r, "right", 0.0, lift_speed, grip_hold):
        return
    for pan in pans:
        r.log.info("[right] 桌面抓取：pan 偏移 %+.0f°", pan)
        if _table_pick_vertical(r, "right", float(pan), lift_speed, grip_hold):
            return
        r.log.warning("[right] pan %+.0f° 没抓住", pan)
        _retreat(r, "right", ["table_hover", "table_approach", "carry"],
                 "right 离开桌面")
    r.log.warning("[right] 标定位姿都没抓住，退回严格视觉伺服")
    try:
        if servo_table_strict(r, "right", lift_speed, grip_hold, tol_px=tol_px):
            return
    except C.StopRequested:
        raise
    except Exception as exc:
        r.log.warning("[right] 严格伺服异常(%s: %s)，继续兜底",
                      type(exc).__name__, exc)
        _retreat(r, "right", ["table_hover", "table_approach", "carry"],
                 "right 离开桌面")
    r.log.warning("[right] 严格伺服没夹住，退回“放桌面”路径反向下探")
    if blind_pick_from_table(r, "right", extra_depths=(0.0,),
                             lift_speed=lift_speed, grip_hold_s=grip_hold,
                             pan_off=0.0, exact=False):
        return
    _retreat(r, "right", ["table_hover", "table_approach", "carry"],
             "right 离开桌面")
    r.log.warning("[right] 盲抓都没成功，改走主循环视觉流程")
    r.phase_pick_from_table("right")


def _align_hover_2dof(r, arm: str, target, hover_key: str,
                      tol_x: float = 15.0, tol_y: float = 18.0,
                      iters: int = 2, probe_deg: float = 6.0):
    """在 hover 位把方块对到 target 像素，返回对齐后的 hover 位姿。

    现场实测（右臂）：pan 管横向约 −3.0px/°、elbow 管纵深约 −5.5px/°，
    增益每次现测（重标后配置里的旧值会让对准过冲、来回震荡）。
    看得见方块才做；看不见返回 None，由调用方走标定位姿。
    """
    if target is None:
        return None
    tx = float(np.ravel(target)[0])
    ty = float(np.ravel(target)[1]) if np.ravel(target).size > 1 else 200.0
    roi = r.box_roi[arm]
    base = r.planned_q(arm, hover_key).copy()
    _, det = r.detect_at(arm, roi=roi, force=True)
    if det is None:
        # 一次没看到就先扫一遍（方块常落在盒子远端/边缘，默认视角看不到）
        r.log.warning("[%s] %s 处没看到方块，扫描一遍", arm, hover_key)
        _, det = r.scan_for_cube(arm, hover_key, roi=roi)
        if det is None:
            r.log.warning("[%s] %s 处完全看不到方块，放弃对准", arm, hover_key)
            return None
        base = r.planned_q(arm, hover_key).copy()
    if abs(float(det.cy) - ty) > 140.0:
        r.log.warning("[%s] %s 处检测 y=%.0f 偏离目标 %.0f，判为不可靠",
                      arm, hover_key, det.cy, ty)
        return None

    gains = {}
    for idx, key, axis in ((0, "pan", "cx"), (2, "elbow", "cy")):
        cand = float(np.clip(base[idx] + probe_deg, -99.0, 99.0))
        if abs(cand - base[idx]) < 0.5:
            r.log.warning("[%s] %s 已到限位，无法量增益", arm, key)
            return None
        q = base.copy()
        q[idx] = cand
        r.move(arm, [q], gripper=r.open_g[arm],
               label=f"{arm} 量 {key} 增益", speed=0.8, precision=True,
               timeout=5.0)
        _, d2 = r.detect_at(arm, roi=roi, force=True)
        r.move(arm, [base], gripper=r.open_g[arm],
               label=f"{arm} 回观察位", speed=0.8, precision=True, timeout=5.0)
        if d2 is None:
            r.log.warning("[%s] 量 %s 增益时看不到方块", arm, key)
            return None
        gains[key] = (float(getattr(d2, axis)) - float(getattr(det, axis))) / (
            cand - base[idx])
    r.log.info("[%s] 实测像素增益: pan %.2fpx/°, elbow %.2fpx/°",
               arm, gains["pan"], gains["elbow"])
    if abs(gains["pan"]) < 0.6 or abs(gains["elbow"]) < 0.6:
        r.log.warning("[%s] 像素增益太小，放弃对准", arm)
        return None

    for it in range(iters):
        _, det = r.detect_at(arm, roi=roi, force=True)
        if det is None:
            break
        ex = float(det.cx) - tx
        ey = float(det.cy) - ty
        r.log.info("[%s] %s 对准第 %d 次: c=(%.0f,%.0f) 误差=(%+.0f,%+.0f)",
                   arm, hover_key, it + 1, det.cx, det.cy, ex, ey)
        if abs(ex) <= tol_x and abs(ey) <= tol_y:
            break
        base[0] = float(np.clip(
            base[0] + np.clip(-ex / gains["pan"], -20.0, 20.0), -99.0, 99.0))
        base[2] = float(np.clip(
            base[2] + np.clip(-ey / gains["elbow"], -9.0, 9.0), -99.0, 99.0))
        r.move(arm, [base], gripper=r.open_g[arm], label=f"{arm} {hover_key} 对准",
               speed=0.8, precision=True, timeout=5.0)
    return base


def _grab_table_at(r, arm: str, base_q, lift_speed: float,
                   grip_hold: float, extra_lift: float = 0.0) -> bool:
    """从给定 hover 位姿竖直下探抓桌面方块（只改 lift/elbow/wrist）。

    extra_lift 是相对竖直增量的额外深度（正数=更深）。vertical_delta 里
    已经扣掉了 table_pick_height_offset（2°，安全余量）；实测按这个余量
    下探两指常常停在方块上方夹空，所以标定抓取时把这点余量补回去，
    正好落到手标 table_pick 深度。
    """
    r.acquire(arm, "table")
    old_hold = r.cfg["grasp"].get("grip_hold_s", 0.15)
    r.cfg["grasp"]["grip_hold_s"] = float(grip_hold)
    try:
        r.move(arm, ["carry", "table_approach"], gripper=r.open_g[arm],
               label=f"{arm} 去桌面", speed=1.0, timeout=5.0)
        hover = np.clip(np.asarray(base_q, dtype=float), -100.0, 100.0).copy()
        r.move(arm, [hover], gripper=r.open_g[arm], label=f"{arm} 桌面观察位",
               speed=0.9, timeout=4.0)
        delta = r.vertical_delta[(arm, "table_hover")].copy()
        if abs(extra_lift) > 1e-6:
            delta[1] += float(extra_lift)
        target = np.clip(hover + delta, -100.0, 100.0)
        r.log.info("[%s] 桌面下探 -> %s", arm, C.fmt_q(target))
        r.move(arm, [target], gripper=r.open_g[arm],
               label=f"{arm} 桌面下探", precision=True, timeout=6.0)
        if r.close_and_verify(arm):
            r.move(arm, ["table_hover"], gripper=r.close_g[arm],
                   label=f"{arm} 夹起", speed=lift_speed, timeout=8.0)
            if r.lift_verify(arm):
                r.move(arm, ["carry"], gripper=r.close_g[arm],
                       label=f"{arm} 去搬运位", speed=1.0, timeout=4.0)
                return True
            r.log.warning("[%s] 桌面抓取抬起滑脱", arm)
        r.release_gripper(arm)
        return False
    finally:
        r.cfg["grasp"]["grip_hold_s"] = old_hold
        r.release(arm, "table")


def _table_pick_vertical(r, arm: str, pan_off: float, lift_speed: float,
                         grip_hold: float) -> bool:
    """在 table_pick + pan_off 处竖直下探抓取。

    先在观察位把 pan 对到抓取点，再只改 lift/elbow/wrist 竖直下去
    （blind_pick_from_table(exact=True) 是斜着从 table_hover 走到
    table_pick：两点的 pan 本来就差 6.8°，夹爪会先扫过方块再落下，抓到
    边缘、方块也被推偏）。
    深度用 vertical_delta（已扣掉 table_pick_height_offset），直接下发
    手标 table_pick 位姿会多下去 2°。
    """
    q_hover = r.pose(arm, "table_hover").copy()
    q_hover[0] = float(r.pose(arm, "table_pick")[0]) + float(pan_off)
    extra = 0.0
    try:
        extra = float(C.arm_cfg(r.cfg, arm).get("table_pick_height_offset", 0.0))
    except Exception:
        extra = 0.0
    return _grab_table_at(r, arm, q_hover, lift_speed, grip_hold,
                          extra_lift=extra)


def pick_from_table_left(r, pans, lift_speed: float, grip_hold: float) -> None:
    """左臂从桌面抓起方块：table_pick 竖直下探 -> pan 扫描 -> 视觉流程。"""
    try:
        if _regrasp_after_place(r, "left", lift_speed, grip_hold):
            return
    except C.StopRequested:
        raise
    except Exception as exc:
        r.log.warning("[left] 原地合爪异常(%s: %s)，继续标定路径",
                      type(exc).__name__, exc)
    r.log.info("[left] 桌面抓取：按 table_pick 位姿竖直下探")
    if _table_pick_vertical(r, "left", 0.0, lift_speed, grip_hold):
        return
    r.log.warning("[left] 标定位姿没夹住，按 pan 偏移扫描")
    _retreat(r, "left", ["table_hover", "table_approach", "carry"], "left 离开桌面")
    for pan in pans:
        r.log.info("[left] 桌面抓取：pan 偏移 %+.0f°", pan)
        if _table_pick_vertical(r, "left", float(pan), lift_speed, grip_hold):
            return
        r.log.warning("[left] pan %+.0f° 没抓住", pan)
        _retreat(r, "left", ["table_hover", "table_approach", "carry"],
                 "left 离开桌面")
    # 最后退回主循环的视觉流程（align + 小范围扫描 + 重试），失败会抛异常。
    r.log.warning("[left] 盲抓都没成功，改走视觉流程")
    r.phase_pick_from_table("left")


def _firm_grip(r, arm: str, pos: float, load: float,
               min_pos_delta: float = 0.8, min_load: float = 140.0) -> bool:
    """闭爪后判断是否真的夹住了方块，而不是只搭到顶棱。

    实测：只搭住顶棱时 pos 只比空夹高一点点，靠负载能过闭爪判定，但一
    抬升就滑；滑脱过程还会把方块拖到盒子另一侧。真夹住时 pos/负载都明显
    更高。不满足就原地松爪换角度，不要抬升。
    """
    st = r.state[arm]
    base = st.air_closed if st.air_closed is not None else float(st.load_thresh)
    return (float(pos) - float(base)) >= min_pos_delta or float(load) >= min_load


def _align_box_xy(r, arm: str, tol_x: float = 15.0, tol_y: float = 18.0):
    """在盒口把方块对到 box_center_px（pan 横向 + elbow 纵深）。

    主循环的盒内对准只调 pan、不调纵深；方块在盒里的纵深位置每次都不同
    （实测 cy 在 154~240 之间），偏 40px 时竖直下探就永远够不到。
    具体实现和增益测量见 _align_hover_2dof。
    """
    target = r.box_center_px.get(arm)
    if target is None:
        return None
    return _align_hover_2dof(r, arm, target, "box_hover",
                             tol_x=tol_x, tol_y=tol_y)


def _pick_box_aligned(r, arm: str, rolls, depth: float, lift_speed: float,
                      grip_hold: float) -> bool:
    """右臂盒内抓取：照左臂视觉流程的结构，换成右臂自己的参数。

    左臂那套（主循环 _visual_pick_from_box）是：盒口检测 -> 横向对准 ->
    竖直下探 -> 合爪验证 -> 夹不稳就加深再试。左臂能直接用是因为方块就落在
    它自己的标定位姿上；右臂的标定位姿与左臂放下的落点有偏差，而且方块在
    盒里的纵深位置每次不同，所以这里把"横向对准"扩成二维：
        pan   -> 横向（实测约 −2.9px/°）
        elbow -> 纵深（实测约 −5.4px/°）
    增益现场实测，不用配置里的旧值（重标后配置是 −2.15，按它算会过冲）。
    下探仍然只改 lift/elbow/wrist，保持竖直；夹得不实就原地松爪，绝不抬升
    （抬升会把方块拖走），再换腕转 / 加深重试。
    """
    r.acquire(arm, "box")
    old_hold = r.cfg["grasp"].get("grip_hold_s", 0.15)
    r.cfg["grasp"]["grip_hold_s"] = float(grip_hold)
    try:
        r.log.info("[%s] 阶段：盒内抓出（视觉对准 + 腕转/深度扫描）", arm)
        r.move(arm, ["carry", "box_approach"], gripper=r.open_g[arm],
               label=f"{arm} 去盒子上方", speed=1.0, timeout=5.0)
        r.move(arm, ["box_hover"], gripper=r.open_g[arm],
               label=f"{arm} 盒内观察位", speed=0.85, timeout=4.0)
        base0 = _align_box_xy(r, arm)
        if base0 is None:
            return False
        delta0 = r.vertical_delta[(arm, "box_hover")].copy()
        delta0[1] += float(depth)
        deepen = float(r.cfg["grasp"].get("box_retry_deepen", 1.2))
        for attempt, extra in enumerate((0.0, deepen, 2.0 * deepen)):
            for roll in rolls:
                q = base0.copy()
                q[4] = float(np.clip(q[4] + float(roll), -100.0, 100.0))
                r.move(arm, [q], gripper=r.open_g[arm],
                       label=f"{arm} 盒内位姿 腕转 {roll:+.0f}", speed=0.9,
                       timeout=4.0)
                target = np.clip(q + delta0 + [0.0, extra, 0.0, 0.0, 0.0],
                                 -100.0, 100.0)
                r.log.info("[%s] 盒内下探（对准后 第%d轮 加深%+.1f）：腕转 %+.0f° -> %s",
                           arm, attempt + 1, extra, roll, C.fmt_q(target))
                r.move(arm, [target], gripper=r.open_g[arm],
                       label=f"{arm} 盒内下探", precision=True, timeout=6.0,
                       arrive_tol=6.0)
                grasped = r.close_and_verify(arm)
                pos = float(r.arms[arm].read_gripper())
                load = float(r.arms[arm].read_load())
                if grasped and _firm_grip(r, arm, pos, load):
                    r.move(arm, ["box_hover"], gripper=r.close_g[arm],
                           label=f"{arm} 夹出", speed=lift_speed, timeout=8.0)
                    if r.lift_verify(arm):
                        r.move(arm, ["box_approach", "carry"],
                               gripper=r.close_g[arm],
                               label=f"{arm} 取出到搬运位", speed=1.0,
                               timeout=5.0)
                        return True
                    r.log.warning("[%s] 腕转 %+.0f° 抬起滑脱", arm, roll)
                else:
                    r.log.warning("[%s] 腕转 %+.0f° 夹得不实（pos=%.2f load=%.0f），"
                                  "原地松爪换角度", arm, roll, pos, load)
                r.release_gripper(arm)
        return False
    finally:
        r.cfg["grasp"]["grip_hold_s"] = old_hold
        r.release(arm, "box")


def place_on_table_at_pick(r, arm: str, hold_for_regrasp: bool = False) -> None:
    """把方块放到桌面，落点对齐到该臂自己的标定抓取点。

    主循环 phase_place_on_table 用 table_hover 的 pan 松爪，而抓取用的是
    table_pick 的 pan：右臂差 12.7°，实测方块正好落在盒子边沿、滚进盒里，
    于是第 3 步"从桌面抓取"扑空。这里松爪前把 pan 对到 table_pick。

    松爪改成"一步步慢开"，并且 hold_for_regrasp=True 时松完就地保持不动——
    下一步原地合爪即可重新夹住，方块根本没机会被碰走。实测"放下再重新
    下探"经常差几毫米夹空，原地合爪没有这个问题。
    """
    r.acquire(arm, "table")
    try:
        r.log.info("[%s] 阶段：放回桌面（落点对齐抓取点）", arm)
        r.move(arm, ["carry", "table_approach"], gripper=r.close_g[arm],
               label=f"{arm} 去桌面", speed=1.0, timeout=5.0)
        r.move(arm, ["table_hover"], gripper=r.close_g[arm],
               label=f"{arm} 桌面释放上方", speed=0.9, timeout=4.0)
        base = r.planned_q(arm, "table_hover").copy()
        base[0] = float(r.pose(arm, "table_pick")[0])
        # 释放深度和抓取深度保持一致：把 table_pick_height_offset 那点余量补回去
        try:
            extra = float(C.arm_cfg(r.cfg, arm).get("table_pick_height_offset", 0.0))
        except Exception:
            extra = 0.0
        drop = r.drop_delta[arm + "_table"].copy()
        if abs(extra) > 1e-6:
            drop[1] += extra
        r.descend_to(arm, base, drop,
                     gripper=r.close_g[arm], label=f"{arm} 竖直放下")
        # 慢开爪：一下弹开会把方块带偏（实测偏移足以让下一次下探夹空）
        a = r.arms[arm]
        open_g = float(r.open_g[arm])
        cur = float(a.read_gripper())
        while cur < open_g - 1.0:
            cur = min(open_g, cur + 8.0)
            a.set_gripper(cur, settle=0.12)
        a.set_gripper(open_g, settle=float(
            r.cfg["motion"].get("release_dwell_s", 0.6)))
        r.state[arm].holding = False
        frame, det = r.detect_at(arm, force=True)
        if det is not None:
            r.log.info("[%s] 桌面确认到方块: %s", arm, det)
        else:
            r.log.warning("[%s] 桌面上没看到方块（下一步抓取会再确认）", arm)
        if hold_for_regrasp:
            _REGRASP_READY[arm] = True
            _REGRASP_POSE[arm] = base.copy()
            r.log.info("[%s] 保持原位，等下一步原地合爪", arm)
            return
        _REGRASP_READY[arm] = False
        r.move(arm, ["table_hover"], gripper=open_g,
               label=f"{arm} 松爪抬起", speed=0.9, timeout=4.0)
        r.move(arm, ["table_approach", "carry", "home"], gripper=open_g,
               label=f"{arm} 回 home", speed=1.0, timeout=6.0)
    finally:
        r.release(arm, "table")


def pick_from_box(r, arm: str, rolls, depth: float, lift_speed: float,
                  grip_hold: float, prefer_visual: bool, note: str = "") -> None:
    """从盒子里夹出方块：右臂先做视觉对准（横向+纵深），再腕转扫描。

    左臂那套盲抓（标定 pan + 腕转 0/±15°）对左臂自己的落点够用，直接沿用；
    右臂的标定落点与左臂放进来的位置有偏差、且纵深每次不同，所以先对准。
    都不行再交给主循环 phase_pick_from_box（视觉流程）。
    """
    if prefer_visual:
        r.phase_pick_from_box(arm, attempt_note=note)
        return
    if arm == "right":
        try:
            if _pick_box_aligned(r, "right", rolls, depth, lift_speed,
                                 grip_hold):
                return
            r.log.warning("[right] 对准抓取没成功，改走腕转扫描")
        except C.StopRequested:
            raise
        except Exception as exc:
            r.log.warning("[right] 对准抓取异常(%s: %s)，改走腕转扫描",
                          type(exc).__name__, exc)
    for roll in rolls:
        r.log.info("[%s] 盒内抓取：腕转 %+.0f°%s", arm, roll, note)
        if blind_pick_from_box(r, arm, extra_depths=(float(depth),),
                               lift_speed=lift_speed, grip_hold_s=grip_hold,
                               roll_deg=roll):
            return
        r.log.warning("[%s] 腕转 %+.0f° 没抓住，换下一个角度", arm, roll)
        _retreat(r, arm, ["box_hover", "box_approach", "carry"],
                 f"{arm} 退出盒子")
    r.log.warning("[%s] 盲抓所有腕转角度都失败，改走视觉流程", arm)
    r.phase_pick_from_box(arm, attempt_note=note)


def home_both(r):
    for a in ("left", "right"):
        try:
            r.move(a, ["home"], gripper=r.open_g[a],
                   label=f"{a} 回 home", speed=1.0, timeout=6.0)
        except C.StopRequested:
            raise
        except Exception as exc:
            r.log.warning("[%s] 回 home 未完成: %s", a, exc)


def _run_single(r, args, logger) -> None:
    """单轮调试模式：保留 --start-box / --stop-after 的语义。"""
    lift_speed = float(args.lift_speed)
    grip_hold = float(args.grip_hold)
    stop_after = int(args.stop_after)
    head_snapshot("cycle_before")

    if args.start_box:
        logger.info("跳过第 1 步（假定方块已在盒内）")
    elif stop_after >= 1:
        logger.info("第 1 步：左臂夹起桌面上的绿色方块放入盒内")
        pick_from_table_left(r, floats(args.left_pans), lift_speed, grip_hold)
        logger.info("[left] 已从桌面夹起方块")
        head_snapshot("cycle_1_picked")
        r.phase_place_in_box("left")
        logger.info("[left] 方块已放入盒子")
        head_snapshot("cycle_1_in_box")

    if stop_after >= 2:
        logger.info("第 2 步：右臂从盒子里夹出方块，放到桌面")
        pick_from_box(r, "right", floats(args.box_rolls),
                      float(args.box_depth), lift_speed, grip_hold,
                      bool(args.visual_box), note="（取出）")
        logger.info("[right] 已把方块从盒子里取出")
        head_snapshot("cycle_2_out_of_box")
        place_on_table_at_pick(r, "right", hold_for_regrasp=True)
        logger.info("[right] 方块已放到桌面")
        head_snapshot("cycle_2_on_table")

    if stop_after >= 3:
        logger.info("第 3 步：右臂从桌面夹起方块，放回盒中")
        pick_from_table_right(r, floats(args.right_pans), lift_speed,
                              grip_hold, tol_px=float(args.table_tol))
        logger.info("[right] 已从桌面夹起方块")
        head_snapshot("cycle_3_picked_from_table")
        r.phase_place_in_box("right")
        logger.info("[right] 方块已放回盒子")
        head_snapshot("cycle_3_back_in_box")

    if stop_after >= 4:
        logger.info("第 4 步：左臂把盒子内的方块拿出来")
        pick_from_box(r, "left", floats(args.box_rolls),
                      float(args.box_depth_left), lift_speed, grip_hold,
                      bool(args.visual_box), note="（取出）")
        logger.info("[left] 已把方块从盒子里拿出来")
        head_snapshot("cycle_4_out_of_box")
        if not args.keep_in_hand:
            logger.info("收尾：左臂把方块放到桌面（避免退出时又放回盒内）")
            place_on_table_at_pick(r, "left")
            logger.info("[left] 方块已放到桌面")
            head_snapshot("cycle_done")

    home_both(r)
    if args.keep_in_hand and stop_after >= 4:
        logger.info("方块留在左爪（退出时夹爪会松开）")


def run_one_cycle(r, args, logger, idx: int) -> None:
    """跑一轮四步循环。失败抛异常，由调用方决定恢复还是收尾。"""
    t0 = time.perf_counter()
    lift_speed = float(args.lift_speed)
    grip_hold = float(args.grip_hold)
    logger.info("=" * 60)
    logger.info("# 第 %d 轮：桌面 -> 盒内 -> 右臂取出放桌 -> 右臂从桌放回 -> 左臂取出", idx)
    head_snapshot(f"cycle{idx}_before")

    # ---- 1. 左臂：桌面 -> 盒子 ----
    if not args.start_box:
        logger.info("第 1 步：左臂夹起桌面上的绿色方块放入盒内")
        pick_from_table_left(r, floats(args.left_pans), lift_speed, grip_hold)
        logger.info("[left] 已从桌面夹起方块")
        head_snapshot("cycle_1_picked")
        r.phase_place_in_box("left")
        logger.info("[left] 方块已放入盒子")
        head_snapshot("cycle_1_in_box")

    # ---- 2. 右臂：盒内取出 -> 放到桌面 ----
    logger.info("第 2 步：右臂从盒子里夹出方块，放到桌面")
    pick_from_box(r, "right", floats(args.box_rolls), float(args.box_depth),
                  lift_speed, grip_hold, bool(args.visual_box), note="（取出）")
    logger.info("[right] 已把方块从盒子里取出")
    head_snapshot("cycle_2_out_of_box")
    place_on_table_at_pick(r, "right", hold_for_regrasp=True)
    logger.info("[right] 方块已放到桌面")
    head_snapshot("cycle_2_on_table")

    # ---- 3. 右臂：从桌面抓取 -> 放回盒中 ----
    logger.info("第 3 步：右臂从桌面夹起方块，放回盒中")
    pick_from_table_right(r, floats(args.right_pans), lift_speed, grip_hold,
                          tol_px=float(args.table_tol))
    logger.info("[right] 已从桌面夹起方块")
    head_snapshot("cycle_3_picked_from_table")
    r.phase_place_in_box("right")
    logger.info("[right] 方块已放回盒子")
    head_snapshot("cycle_3_back_in_box")

    # ---- 4. 左臂：盒内取出 -> 放回桌面 ----
    logger.info("第 4 步：左臂把盒子内的方块拿出来")
    pick_from_box(r, "left", floats(args.box_rolls),
                  float(args.box_depth_left), lift_speed, grip_hold,
                  bool(args.visual_box), note="（取出）")
    logger.info("[left] 已把方块从盒子里拿出来")
    head_snapshot("cycle_4_out_of_box")
    if not args.keep_in_hand:
        logger.info("收尾：左臂把方块放到桌面（下一轮的起点）")
        place_on_table_at_pick(r, "left")
        logger.info("[left] 方块已放到桌面")
        head_snapshot("cycle_done")

    home_both(r)
    logger.info("# 第 %d 轮完成，用时 %.1fs", idx, time.perf_counter() - t0)


def recover_to_start(r, logger) -> None:
    """一轮失败后尽力把现场恢复到"方块在桌面、两臂在 home"。

    失败常见在右臂桌面点（方块落在盒子右边、再抓空），这时下一轮左臂
    够不到它。所以恢复分两步：
      1) 右臂在自己桌面点试探抓一次，抓到就放回盒子；
      2) 左臂把盒里的方块取出来放到自己的桌面点（下一轮的第 1 步起点）。
    全程用短路径（每步只试一两次），别让一次恢复吃掉半分钟以上。
    """
    def _home(a: str) -> None:
        _raise_clear(r, a, label="恢复抬离")
        try:
            r.move(a, ["home"], gripper=r.open_g[a],
                   label=f"{a} 恢复回 home", speed=1.0, timeout=10.0)
        except C.StopRequested:
            raise
        except Exception as exc:
            logger.warning("[%s] 恢复回 home 未完成: %s", a, exc)

    for a in ("left", "right"):
        _home(a)
    _REGRASP_READY.clear()
    _REGRASP_POSE.clear()
    logger.warning("恢复：先看右臂桌面点有没有方块")
    try:
        picked = (_table_pick_vertical(r, "right", 0.0, 0.35, 0.3)
                  or _table_pick_vertical(r, "right", 8.0, 0.35, 0.3)
                  or _table_pick_vertical(r, "right", -8.0, 0.35, 0.3))
        if picked:
            r.phase_place_in_box("right")
            logger.info("恢复：方块已从右侧桌面放回盒子")
    except C.StopRequested:
        raise
    except Exception as exc:
        logger.warning("恢复：右臂桌面抓取未成功（%s: %s）",
                       type(exc).__name__, exc)
        _home("right")
    logger.warning("恢复：把盒里的方块取到左臂桌面点")
    try:
        if blind_pick_from_box(r, "left", extra_depths=(0.0,), roll_deg=0.0,
                               lift_speed=0.35, grip_hold_s=0.2):
            place_on_table_at_pick(r, "left")
            logger.info("恢复完成：方块已放回桌面")
            return
        logger.info("恢复：盒里没有方块（或没夹到），保持当前现场")
    except C.StopRequested:
        raise
    except Exception as exc:
        logger.warning("恢复取出未成功: %s: %s", type(exc).__name__, exc)
    for a in ("left", "right"):
        _home(a)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="双臂四步循环：左臂放盒 -> 右臂取出放桌 -> 右臂从桌放回 -> 左臂取出"
                    "（可循环往复）")
    ap.add_argument("--box-depth", type=float, default=0.0,
                    help="右臂盒内下探的额外深度（默认 0，与左臂一致；"
                         "配置的 box_pick_height_offset 已经包含在竖直增量里）")
    ap.add_argument("--box-depth-left", type=float, default=0.0,
                    help="左臂盒内下探的额外深度（默认 0，实测再深会压到盒底）")
    ap.add_argument("--box-rolls", default="0,15,-15",
                    help="盒内抓取依次尝试的腕转角度，逗号分隔")
    ap.add_argument("--left-pans", default="16,-16,24,-24",
                    help="左臂桌面抓取回退时依次尝试的 pan 偏移（度）")
    ap.add_argument("--right-pans", default="20,-16",
                    help="右臂桌面抓取回退时依次尝试的 pan 偏移（度）")
    ap.add_argument("--table-tol", type=float, default=12.0,
                    help="右臂桌面伺服的收敛容差（像素），默认 12")
    ap.add_argument("--start-box", action="store_true",
                    help="跳过第 1 步（假定方块已在盒内），只跑 2~4 步")
    ap.add_argument("--stop-after", type=int, default=4, choices=(1, 2, 3, 4),
                    help="只跑到第 N 步就收尾（调试用）")
    ap.add_argument("--keep-in-hand", action="store_true",
                    help="收尾不把方块放到桌面（方块留在左爪，退出时会松开放下）")
    ap.add_argument("--visual-box", action="store_true",
                    help="盒内抓取先走主循环视觉流程（默认标定+对准优先）")
    ap.add_argument("--lift-speed", type=float, default=0.3,
                    help="抓取后抬升速度倍率")
    ap.add_argument("--grip-hold", type=float, default=0.5,
                    help="合爪后停留时间（秒）")
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="循环执行这么久（分钟）；0=只跑一轮")
    ap.add_argument("--cycles", type=int, default=0,
                    help="循环执行这么多轮；0=不限（配合 --minutes）")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="four_step_cycle")
    sup = Supervisor(cfg, logger)
    sup._signals()

    minutes = float(args.minutes)
    max_cycles = int(args.cycles)
    loop_mode = minutes > 0.0 or max_cycles > 0

    ok = False
    try:
        sup._setup()
        r = sup.runner
        # 盒内检测范围放宽：方块有时落在盒子远端边缘（腕相机画面里 y 很小），
        # 配置里 box_roi 的上边界 60 会把它截掉，右臂就"看不到"方块、只能盲抓。
        for _a in ("left", "right"):
            r.box_roi[_a] = [40, 10, 632, 478]
        t_start = time.time()
        deadline = t_start + minutes * 60.0 if minutes > 0.0 else None
        idx = 0
        n_ok = 0
        while True:
            if max_cycles and idx >= max_cycles:
                break
            if deadline is not None and idx > 0 and time.time() >= deadline:
                break
            idx += 1
            try:
                if not loop_mode:
                    # 单轮调试模式：沿用 --stop-after 的语义
                    _run_single(r, args, logger)
                    n_ok += 1
                else:
                    run_one_cycle(r, args, logger, idx)
                    n_ok += 1
                    logger.info("[轮次] 第 %d 轮成功（成功 %d / 已跑 %d，用时 %.0fs）",
                                idx, n_ok, idx, time.time() - t_start)
            except C.StopRequested:
                raise
            except Exception as exc:
                logger.error("[轮次] 第 %d 轮失败: %s: %s", idx,
                             type(exc).__name__, exc)
                logger.debug(traceback.format_exc())
                head_snapshot(f"cycle{idx}_fail")
                try:
                    recover_to_start(r, logger)
                except C.StopRequested:
                    raise
                except Exception as exc2:
                    logger.error("[轮次] 恢复也失败: %s: %s",
                                 type(exc2).__name__, exc2)
            if not loop_mode:
                break
            if deadline is not None and time.time() >= deadline:
                break
        if loop_mode:
            logger.info("[轮次] 循环结束：成功 %d 轮 / 共 %d 轮，总用时 %.0fs",
                        n_ok, idx, time.time() - t_start)
        home_both(r)
        logger.info("四步循环完成：桌面 -> 盒内 -> 右臂取出放桌 -> 右臂从桌放回 -> 左臂取出")
        ok = n_ok > 0
        return 0 if n_ok > 0 else 1
    except C.StopRequested:
        logger.warning("收到停止请求")
        return 130
    except Exception as exc:
        logger.error("失败: %s: %s", type(exc).__name__, exc)
        logger.debug(traceback.format_exc())
        try:
            if sup.runner is not None:
                for a in ("left", "right"):
                    # 先原地竖直抬离桌面，再横移回 home：张开的夹爪贴着桌面
                    # 横走会把方块扫走（上一次就是这样把方块推进盒子的）。
                    _raise_clear(sup.runner, a, label="退出抬离")
                    sup.runner.move(a, ["home"],
                                    gripper=sup.runner.open_g[a],
                                    label=f"{a} 退出回 home", speed=1.0,
                                    timeout=10.0)
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
