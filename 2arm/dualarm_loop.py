#!/usr/bin/env python
"""dualarm_loop.py — 双臂方块循环夹放（YOLO + 死编程位姿，24/7）。

动作流程（一轮）：
    1. 左臂：桌面抓绿色方块 -> 放入盒子
    2. 右臂：从盒子里把方块夹出来（抬到盒口上方）
    3. 右臂：把方块放回盒中
    4. 左臂：把盒内方块夹出来 -> 放回桌面原位
    然后重复 1~4。

说明：
    第 4 步末尾“放回桌面原位”是为了闭合循环——不这样的话下一轮左臂
    在桌面上就找不到方块。如果你想要别的收尾（例如放在固定托盘），
    只要把 left 的 `table_drop` 位姿标到那里即可，程序不用改。

设计原则：
    * 全部关键点位来自标定，视觉只做“确认 + 小幅修正”，不做长时间伺服；
    * 每条臂一个后台轨迹线程，100Hz 绝对时基发送，两条臂可并行；
    * 夹爪在轨迹中开合（释放时边升边松），减少“停下-等待-再走”的停顿；
    * 盒子区域同一时刻只允许一条臂进入（互锁），避免互撞；
    * 任何异常都自动恢复并继续，进程崩溃由 systemd 拉起。

用法（conda activate hupy3.12）：
    python dualarm_loop.py self-test              # 只读体检，不动电机
    python dualarm_loop.py preview --arm left --seconds 10
    python dualarm_loop.py run --cycles 1         # 先跑一轮，人守着急停
    python dualarm_loop.py run --cycles 0         # 无限循环（24/7）
    python dualarm_loop.py run --cycles 0 --seconds 300   # 跑 5 分钟
    python dualarm_loop.py home                   # 两臂回 home
停止：Ctrl+C（会先开爪、再下电）。
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import armcore as C
from armdriver import DualArm, probe_port_counts, resolve_arm_ports
from armvision import (
    CameraError,
    CameraStream,
    Detection,
    Detector,
    VisionAligner,
    box_roi_from_cfg,
    save_snapshot,
)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def cam_device(cfg: dict, arm: str) -> str:
    """优先按 USB 端口找相机，其次才用写死的设备路径。

    /dev/videoN 在插拔后会重新编号，写死路径会中途指向另一台相机。
    """
    d = cfg["devices"]
    usb = d.get(f"{arm}_camera_usb")
    if usb:
        got = C.find_video_by_usb(str(usb))
        if got:
            return got
    primary = str(d.get(f"{arm}_camera") or "")
    for cand in [primary] + list(d.get(f"{arm}_camera_fallbacks") or []):
        if cand and Path(str(cand)).exists():
            return str(cand)
    return primary


class PhaseTimer:
    """阶段超时保护：超时抛 PhaseError，由上层重试/恢复。"""

    def __init__(self, seconds: float, name: str):
        self.seconds = float(seconds)
        self.name = name
        self.t0 = time.perf_counter()

    def check(self) -> None:
        if self.seconds > 0 and time.perf_counter() - self.t0 > self.seconds:
            raise C.PhaseError(f"阶段 {self.name} 超时 ({self.seconds:.0f}s)")

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    @property
    def remaining(self) -> float:
        if self.seconds <= 0:
            return float("inf")
        return self.seconds - self.elapsed


class CycleCSV:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header = ["ts", "cycle", "success", "duration_s", "phase", "error"]

    def write(self, cycle: int, ok: bool, duration: float, phase: str, err: str):
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(self.header)
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), cycle, int(ok),
                        round(duration, 2), phase, err[:300]])


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

@dataclass
class ArmState:
    air_closed: Optional[float] = None
    load_available: bool = False
    load_thresh: float = 0.0
    holding: bool = False
    last_zone: Optional[str] = None


class LoopRunner:
    def __init__(self, cfg: dict, dual: DualArm, cams: dict[str, CameraStream],
                 detector: Detector, stop_event: threading.Event, logger,
                 csv_writer: Optional[CycleCSV] = None):
        self.cfg = cfg
        self.dual = dual
        self.arms = dual.arms
        self.cams = cams
        self.det = detector
        self.stop_event = stop_event
        self.log = logger
        self.csv = csv_writer

        self.state = {a: ArmState() for a in ("left", "right")}
        self.aligners = {a: VisionAligner(cfg, a, detector, logger) for a in ("left", "right")}

        # 位姿缓存（只读）
        self.P = {a: {k: C.get_pose(cfg, a, k) for k in C.POSE_KEYS}
                  for a in ("left", "right")}
        self.open_g = {a: float(C.arm_cfg(cfg, a)["gripper_open"]) for a in ("left", "right")}
        self.close_g = {a: float(C.arm_cfg(cfg, a)["gripper_closed"]) for a in ("left", "right")}
        self.table_offset = {a: float(C.arm_cfg(cfg, a).get("table_pick_height_offset") or 0.0)
                             for a in ("left", "right")}
        self.box_offset = {a: float(C.arm_cfg(cfg, a).get("box_pick_height_offset") or 0.0)
                           for a in ("left", "right")}
        self.box_roi = {a: box_roi_from_cfg(cfg, a) for a in ("left", "right")}

        self.zone_owner: dict[str, Optional[str]] = {"table": None, "box": None}
        self._zone_lock = threading.Lock()
        self._placing_retry = {"left": False, "right": False}
        # 高温降速倍率，_check_temp 每轮更新
        self.speed_factor = 1.0
        self.cycle = 0
        self.success = 0
        self.failures = 0

        # 竖直下探的 “hover -> pick” 关节增量（只保留 lift/elbow/wrist）
        self.vertical_delta = {}
        for a in ("left", "right"):
            for hk, pk in (("table_hover", "table_pick"), ("box_hover", "box_pick")):
                d = self.P[a][pk] - self.P[a][hk]
                vd = d.copy()
                vd[0] = 0.0  # pan
                vd[4] = 0.0  # roll
                self.vertical_delta[(a, hk)] = vd
        for a in ("left", "right"):
            for (aa, hk), vd in self.vertical_delta.items():
                if aa != a:
                    continue
                off = self.box_offset[a] if hk == "box_hover" else self.table_offset[a]
                if abs(off) > 1e-6:
                    # 约定：shoulder_lift 增大 = 更低，所以“抬高 off”要减
                    vd[1] -= off
        self.drop_delta = {}
        for a in ("left", "right"):
            self.drop_delta[a] = (self.P[a]["box_drop"] - self.P[a]["box_hover"]).copy()
            self.drop_delta[a][0] = 0.0
            self.drop_delta[a][4] = 0.0
            # 正数 = 在 box_drop 之上再抬一些（shoulder_lift 减小）
            # 释放位必须跟抓取位用同一个高度余量：box_drop 本来就比 box_pick
            # 深（右臂 +2°，左臂 +7°），只抬高抓取不抬高释放，夹爪松爪那一
            # 刻就会压到盒底/桌面——就是“机械爪撞到桌子”的来源。
            self.drop_delta[a][1] -= float(self.box_offset[a])
            self.drop_delta[a][1] -= float(self.cfg["grasp"].get("box_drop_extra_lift", 0.0))
        for a in ("left", "right"):
            # 桌面释放高度：pick 位再抬高一点，方块落下几毫米
            td = self.vertical_delta[(a, "table_hover")].copy()
            td[1] -= float(self.cfg["grasp"].get("table_release_lift", 0.0))
            self.drop_delta[a + "_table"] = td
        # 落点目标像素（标定 box_hover 时盒内方块的像素位置），以及不重标
        # 就能左右/前后微调的偏移。落点验证靠它们。
        self.drop_px = {}
        self.drop_offset = {}
        # 盒内抓取的中心像素与 pan 像素增益：观察位把方块横向对到两指中线，
        # 再竖直下探。盒子比桌面深得多，逐段对中会把臂往回抬（见
        # _visual_pick_from_box 的说明），所以只在观察位对一次。
        self.box_center_px = {}
        self.box_px_gain = {}
        for a in ("left", "right"):
            g = C.arm_cfg(self.cfg, a).get("gains") or {}
            px = g.get("drop_px")
            self.drop_px[a] = np.asarray(px, dtype=float) if px else None
            off = g.get("drop_offset") or [0.0, 0.0]
            self.drop_offset[a] = np.asarray(off, dtype=float)
            cpx = g.get("box_center_px")
            self.box_center_px[a] = np.asarray(cpx, dtype=float) if cpx else None
            bpg = g.get("box_pan_px_per_deg")
            self.box_px_gain[a] = float(bpg) if bpg else None

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------
    def _check_stop(self):
        if self.stop_event.is_set():
            raise C.StopRequested()

    def _check_temp(self):
        t = self.dual.max_temp()
        warn = float(self.cfg["safety"]["temp_warn_c"])
        slow = float(self.cfg["safety"].get("temp_slow_c", 68.0))
        stop = float(self.cfg["safety"]["temp_stop_c"])
        if t >= stop:
            self.log.error("电机温度 %.0f°C 达到停机阈值 %.0f°C，卸力冷却 %.0fs",
                           t, stop, float(self.cfg["safety"]["temp_wait_s"]))
            self.dual.disable_torque()
            self._sleep(float(self.cfg["safety"]["temp_wait_s"]))
            self.dual.enable_torque()
            self.speed_factor = 1.0
            self._recover_after_pause()
        elif t >= slow:
            # 长时间连续运行时电机温度会缓慢爬升。降速让电流下来，
            # 避免直接冲到 74°C 停机，比停下来冷却更连贯。
            self.speed_factor = float(self.cfg["motion"].get("temp_slow_scale", 0.7))
            self.log.warning("电机温度 %.0f°C ≥ %.0f°C，降速到 %.0f%%",
                             t, slow, self.speed_factor * 100.0)
        else:
            self.speed_factor = 1.0
            if t >= warn:
                self.log.warning("电机温度偏高 %.0f°C", t)

    def _sleep(self, seconds: float):
        end = time.perf_counter() + max(0.0, seconds)
        while time.perf_counter() < end:
            self._check_stop()
            time.sleep(min(0.1, end - time.perf_counter()))

    def _recover_after_pause(self):
        """冷却/暂停后把两臂摆回 home，并重测空夹值。"""
        for a in ("left", "right"):
            try:
                self.arms[a].start([self.P[a]["home"]], gripper=self.open_g[a],
                                   label=f"{a} 回 home", speed_scale=0.7)
            except Exception as exc:
                self.log.warning("[%s] 回 home 失败: %s", a, exc)
        for a in ("left", "right"):
            try:
                self.arms[a].wait(timeout=6.0)
            except Exception:
                pass
        self._measure_air_closed()

    def _measure_air_closed(self):
        """在 measure 位空夹一次，得到抓取判定的基线。"""
        grasp = self.cfg["grasp"]
        for a in ("left", "right"):
            arm = self.arms[a]
            try:
                arm.start([self.P[a]["measure"]], gripper=self.open_g[a],
                          label=f"{a} 去 measure", speed_scale=0.6)
                arm.wait(timeout=4.0)
                pos = arm.set_gripper(self.close_g[a], settle=float(
                    self.cfg["motion"]["gripper_close_settle_s"]) + 0.25)
                load = arm.read_load()
                st = self.state[a]
                st.air_closed = float(pos)
                st.load_available = 0.5 < load < 300.0
                st.load_thresh = load + float(grasp["load_min"])
                arm.set_gripper(self.open_g[a], settle=float(
                    self.cfg["motion"]["gripper_open_settle_s"]))
                self.log.info("[%s] 空夹 pos=%.2f load=%.1f 负载可用=%s 阈值=%.1f",
                              a, pos, load, st.load_available, st.load_thresh)
                if pos > 20.0:
                    self.log.warning("[%s] 空夹读数偏高（%.1f），measure 位可能被挡", a, pos)
            except C.StopRequested:
                raise
            except Exception as exc:
                self.log.error("[%s] 测空夹失败: %s", a, exc)
                st = self.state[a]
                st.air_closed = float(self.close_g[a])
                st.load_available = False
                st.load_thresh = float(grasp["load_min"])

    # ---- 区域互锁 ----
    def acquire(self, arm: str, zone: str):
        with self._zone_lock:
            owner = self.zone_owner.get(zone)
            if owner is not None and owner != arm:
                raise C.PhaseError(f"{zone} 区域被 {owner} 占用，不能同时进入")
            self.zone_owner[zone] = arm

    def release(self, arm: str, zone: str):
        with self._zone_lock:
            if self.zone_owner.get(zone) == arm:
                self.zone_owner[zone] = None

    # ---- 动作原语 ----
    def pose(self, arm: str, key: str) -> np.ndarray:
        return self.P[arm][key]

    def planned_q(self, arm: str, fallback_key: str = "home") -> np.ndarray:
        """上一条指令的终点（可能尚未执行完），用于续接轨迹。"""
        a = self.arms[arm]
        q = getattr(a, "_planned_q", None)
        if q is None:
            q = a.hold_q
        if q is None:
            q = self.pose(arm, fallback_key)
        return np.asarray(q, dtype=float)

    def move(self, arm: str, keys, *, gripper=None, label="", speed=1.0,
             precision=False, timeout=None, gripper_to=None, grip_switch=0.28,
             arrive_tol=None):
        """按位姿名依次移动；可选在轨迹末段改变夹爪开度。"""
        wps = [self.pose(arm, k) if isinstance(k, str) else np.asarray(k, float)
               for k in keys]
        a = self.arms[arm]
        # 如果已经站在某个中间路点上（预位造成的），从该点开始走，
        # 避免“先退回去再走回来”的多余往返。
        # 用 _planned_q（上一条指令的终点）而不是实际位置：预位是异步的，
        # 提交时臂可能还在飞往预位点的途中。
        ref = getattr(a, "_planned_q", None)
        if ref is None:
            ref = a.hold_q
        if ref is not None and len(wps) > 1:
            start_k = 0
            for k in range(len(wps)):
                if float(np.abs(wps[k] - ref).max()) <= 1.5:
                    start_k = k
            if start_k:
                wps = wps[start_k:]
        tmo = timeout if timeout is not None else max(3.0, 0.35 * len(wps) + 3.0)
        scale = speed * self.speed_factor
        if gripper_to is not None:
            a.start_mixed(wps, gripper_from=gripper, gripper_to=gripper_to,
                          switch_frac=grip_switch, label=label or ",".join(map(str, keys)),
                          speed_scale=scale, precision=precision)
        else:
            a.start(wps, gripper=gripper, label=label or ",".join(map(str, keys)),
                    speed_scale=scale, precision=precision)
        if not a.wait(timeout=tmo, tol=arrive_tol):
            msg = f"{arm} {label} 未在 {tmo:.0f}s 内到位"
            if precision:
                # 下探没到位就合爪，等于在错误高度抓取/释放，可能顶到限位。
                # 以前这里只警告然后继续，是“撞了还继续干”的来源。
                raise C.PhaseError(msg)
            self.log.warning("[%s] %s 未在 %.1fs 内到位，继续", arm, label, tmo)
        self._check_stop()

    def descend(self, arm: str, hover_key: str, *, gripper, label=""):
        """从当前 hover 位竖直下探到标定的抓取/释放高度。

        只把标定好的 lift/elbow/wrist 增量叠加上去，pan/roll 保持当前值，
        所以即使视觉把 hover 平移了一点，下探仍然是竖直的。
        """
        a = self.arms[arm]
        # 用“计划终点”而不是实际位置：视觉对准是同步的，但预位/并行启动
        # 可能让臂还在动，用实际位置会把下探起点带偏。
        try:
            q_cur = self.planned_q(arm, hover_key)
        except Exception:
            q_cur = self.pose(arm, hover_key)
        target = np.asarray(q_cur, dtype=float) + self.vertical_delta[(arm, hover_key)]
        target = np.clip(target, -100.0, 100.0)
        self.move(arm, [target], gripper=gripper, label=label, speed=1.0,
                  precision=True, timeout=6.0)
        return target

    def descend_to(self, arm: str, base_q, delta, *, gripper, label=""):
        target = np.clip(np.asarray(base_q, float) + np.asarray(delta, float), -100, 100)
        self.move(arm, [target], gripper=gripper, label=label, speed=1.0,
                  precision=True, timeout=6.0)
        return target

    def descend_tracking(self, arm: str, hover_key: str, delta, *, gripper,
                         roi=None, steps: int = 4, label: str = "",
                         arrive_tol: float = 6.0, strict: bool = False,
                         track: bool = True, base=None) -> np.ndarray:
        """分段下探，可选每段用腕相机把目标重新对中。

        track=True 只适合浅下探（桌面那套，lift 增量约 20°）：夹爪下降时
        会把方块挤到一边，每段重新对中能让方块留在两指之间。
        深下探（盒内 lift 增量约 48°）不能用：相机跟着臂一起下降，方块
        在画面里的纵向像素本来就会一路漂移（实测 y 202->330，
        dy/dshoulder_lift ≈ -5.7 px/度），按 hover 时刻的像素对中等于
        让对准器一路把臂往回抬，最后合爪位置远高于方块。

        base 是下探起点（默认取 hover_key 的规划位姿）。调用方在观察位
        做过横向对准时要把对准后的位姿传进来，否则那一步 pan 修正会被
        这里重新取回的标定位姿覆盖掉。
        """
        aligner = self.aligners[arm]
        target_px = None
        if track:
            # 目标像素用“下探开始那一刻方块的位置”，而不是 align_px。
            # hover 位本身就是标定好、夹爪正对目标的姿态，此刻的像素才是最
            # 可靠的参考；align_px 是另一次标定留下的，可能对不上。
            _, det0 = self.detect_at(arm, roi=roi, force=True)
            target_px = det0.center if det0 is not None else aligner.target_px()
            if target_px is not None:
                self.log.info("[%s] 下探参考像素 %s", arm, np.round(target_px, 0).tolist())
        q = (np.asarray(base, dtype=float).copy() if base is not None
             else self.planned_q(arm, hover_key).copy())
        step = np.asarray(delta, dtype=float) / max(1, steps)
        # 下探的最大深度：只到标定抓取位，视觉对中不允许把臂再往下压。
        # aligner 会调 shoulder_lift 去把方块拉到画面中心，方块偏低时它会
        # 一路降低——不设上限就会把夹爪压到盒底/桌面上。
        q_deep = np.clip(q + np.asarray(delta, dtype=float), -100.0, 100.0)
        lift_floor = float(q_deep[1])
        for i in range(steps):
            q = np.clip(q + step, -100.0, 100.0)
            if strict:
                self.move(arm, [q], gripper=gripper,
                          label=f"{label} {i + 1}/{steps}", precision=True,
                          timeout=8.0, arrive_tol=arrive_tol)
            else:
                # 盒内下探常被盒底/方块顶住，残差几度是正常的。这时继续
                # 往下压没有意义（关节已经被挡住），要立刻停下——继续走
                # 下一段就是把夹爪往盒底/桌面上按。
                a = self.arms[arm]
                a.start([q], gripper=gripper,
                        label=f"{label} {i + 1}/{steps}",
                        speed_scale=self.speed_factor,
                        precision=True)
                if not a.wait(timeout=8.0, tol=arrive_tol):
                    self.log.warning("[%s] %s %d/%d 残差偏大（容差 %.0f），"
                                     "判定触底，停止下探", arm, label, i + 1,
                                     steps, arrive_tol)
                    break
            if target_px is None:
                continue
            try:
                q, det = aligner.align(self.arms[arm], self.cams[arm], target_px, q, roi=roi)
                # 对中只允许左右/前后修，不许多下一分：对准过程中把
                # shoulder_lift 降低会把夹爪压到盒底/桌面。
                if q[1] > lift_floor:
                    q[1] = lift_floor
                if det is not None:
                    err = float(np.abs(det.center - np.asarray(target_px)).max())
                    self.log.debug("[%s] 下探第 %d 段对准残差 %.0fpx", arm, i + 1, err)
            except Exception as exc:
                self.log.warning("[%s] 下探中对准失败: %s", arm, exc)
        return q

    def close_and_verify(self, arm: str) -> bool:
        a = self.arms[arm]
        st = self.state[arm]
        grasp = self.cfg["grasp"]
        # 分几步合爪。一步从张到合是伺服全速，指端会“啪”地打上去，
        # 轻的方块容易被弹开，看起来像夹住了其实两指之间是空的。
        target = self.close_g[arm]
        cur = float(a.read_gripper())
        step = float(grasp.get("soft_close_step", 8.0))
        while cur > target + 0.5:
            cur = max(target, cur - step)
            a.set_gripper(cur, settle=float(grasp.get("soft_close_settle_s", 0.10)))
        pos = a.set_gripper(target, settle=float(
            self.cfg["motion"]["gripper_close_settle_s"]))
        # 多停一小会再抬：夹爪刚合上时方块还在往指间滑，立刻抬升容易滑脱。
        time.sleep(float(grasp.get("grip_hold_s", 0.15)))
        load = a.read_load()
        # 再等一小会读第二次，取较小值：夹爪顶到盒底/桌面时负载会瞬时冲高
        # （实测 112），一两百毫秒内就掉下来（48），而真夹住方块能一直维持
        # 200 以上。只读一次会被这个瞬时尖峰骗成“夹住”，抬起来才发现是空的。
        time.sleep(float(grasp.get("load_settle_s", 0.25)))
        load = min(load, a.read_load())
        base = st.air_closed if st.air_closed is not None else float(st.load_thresh)
        by_pos = pos > base + float(grasp["pos_margin"])
        by_load = st.load_available and load >= st.load_thresh
        ok = by_pos or by_load
        self.log.info("[%s] 闭爪: pos=%.2f (空夹 %.2f) load=%.1f -> %s",
                      arm, pos, base, load, "夹住" if ok else "空爪")
        st.holding = ok
        return ok

    def lift_verify(self, arm: str) -> bool:
        """抬起后复检是否仍夹住（防止闭爪瞬间夹到、抬起滑脱）。"""
        a = self.arms[arm]
        st = self.state[arm]
        grasp = self.cfg["grasp"]
        pos = a.read_gripper()
        load = a.read_load()
        base = st.air_closed if st.air_closed is not None else float(st.load_thresh)
        by_pos = pos > base + float(grasp["pos_margin"])
        by_load = st.load_available and load >= st.load_thresh
        ok = by_pos or by_load
        self.log.info("[%s] 抬升后: pos=%.2f load=%.1f -> %s",
                      arm, pos, load, "夹住" if ok else "滑脱")
        st.holding = ok
        return ok

    def release_gripper(self, arm: str):
        a = self.arms[arm]
        a.set_gripper(self.open_g[arm], settle=float(
            self.cfg["motion"]["gripper_open_settle_s"]))
        self.state[arm].holding = False

    def release_here(self, arm: str, back_key: str, *, label: str,
                     speed: float = 0.9, timeout: float = 4.0):
        """原地张开夹爪、停一会儿让方块落稳，再抬回 back_key。

        比 lift_releasing 慢约 0.2s，但方块不会被抬起的夹爪带偏，
        每次落点更一致（“盒内固定一点”要求的就是这个）。
        """
        a = self.arms[arm]
        dwell = float(self.cfg["motion"].get("release_dwell_s", 0.18))
        a.set_gripper(self.open_g[arm], settle=dwell)
        self.state[arm].holding = False
        self.move(arm, [back_key], gripper=self.open_g[arm],
                  label=label, speed=speed, timeout=timeout)

    def lift_releasing(self, arm: str, keys, *, label: str, speed: float = 0.9,
                       timeout: float = 4.0):
        """抬起的同时把夹爪从闭合连续打开——省掉一段原地等待。"""
        a = self.arms[arm]
        wps = [self.pose(arm, k) if isinstance(k, str) else np.asarray(k, float)
               for k in keys]
        a.start_mixed(wps, gripper_from=self.close_g[arm],
                      gripper_to=self.open_g[arm], switch_frac=0.25,
                      label=label, speed_scale=speed)
        a.wait(timeout=timeout)
        self.state[arm].holding = False
        self._check_stop()

    # ---- 视觉 ----
    def detect_at(self, arm: str, roi=None, force=False) -> tuple[Optional[np.ndarray],
                                                                 Optional[Detection]]:
        cam = self.cams[arm]
        frame = cam.fresh(max_age=0.8)
        if frame is None:
            try:
                frame = cam.read(timeout=1.0)
            except CameraError as exc:
                self.log.warning("[%s] 相机异常: %s", arm, exc)
                return None, None
        det = self.det.detect(frame, roi=roi, cache_key=arm, force_yolo=force)
        return frame, det

    def align_at_hover(self, arm: str, hover_key: str, roi=None):
        """在 hover 位做小幅视觉对准。返回 (对齐后的位姿, 检测)。"""
        target = self.aligners[arm].target_px()
        base = self.pose(arm, hover_key)
        if target is None:
            frame, det = self.detect_at(arm, roi=roi, force=True)
            return base, det
        q, det = self.aligners[arm].align(self.arms[arm], self.cams[arm], target,
                                         base, roi=roi)
        return q, det

    def scan_for_cube(self, arm: str, hover_key: str, roi=None):
        """找不到方块时按固定 pan 偏置小范围扫一遍（不做长时间伺服）。"""
        base = self.pose(arm, hover_key)
        pan = C.JOINTS.index("shoulder_pan")
        a = self.arms[arm]
        for dp in (0.0, -5.0, 5.0, -9.0, 9.0):
            q = base.copy()
            q[pan] += dp
            if abs(dp) > 1e-6:
                a.start([q], gripper=self.open_g[arm],
                        label=f"{arm} 扫描 {dp:+.0f}", speed_scale=0.9)
                a.wait(timeout=3.0)
            frame, det = self.detect_at(arm, roi=roi, force=True)
            if det is not None:
                self.log.info("[%s] 扫描 pan%+.0f 找到方块: %s", arm, dp, det)
                return q, det
        return None, None

    # ------------------------------------------------------------------
    # 阶段 1：左臂 桌面 -> 盒子
    # ------------------------------------------------------------------
    def phase_pick_from_table(self, arm: str):
        self.acquire(arm, "table")
        try:
            timer = PhaseTimer(float(self.cfg["safety"]["phase_timeout_s"]), f"{arm} 桌面抓取")
            self.log.info("[%s] 阶段：桌面抓取", arm)
            self.move(arm, ["carry", "table_approach"], gripper=self.open_g[arm],
                      label=f"{arm} 去桌面", speed=1.0)
            self.move(arm, ["table_hover"], gripper=self.open_g[arm],
                      label=f"{arm} 桌面观察位", speed=0.9, timeout=4.0)
            timer.check()
            q_hover, det = self.align_at_hover(arm, "table_hover")
            if det is None:
                self.log.warning("[%s] hover 位没看到方块，开始小范围扫描", arm)
                q_hover, det = self.scan_for_cube(arm, "table_hover")
                if det is None:
                    raise C.TargetMissing(f"{arm} 桌面上找不到方块")
            timer.check()
            if not self._align_ok(arm, det) and det is not None:
                self.log.info("[%s] 方块偏离对准点 %.0f px（继续用标定位姿）", arm,
                              float(np.linalg.norm(det.center - self.aligners[arm].target_px()))
                              if self.aligners[arm].target_px() is not None else -1.0)
            pick_q = self.descend(arm, "table_hover", gripper=self.open_g[arm],
                                  label=f"{arm} 竖直下探")
            timer.check()
            grasp_ok = self.close_and_verify(arm)
            if not grasp_ok:
                self.log.warning("[%s] 桌面抓取未夹住，重试", arm)
                self.release_gripper(arm)
                self._retry_table_pick(arm, timer)
                return
            self.move(arm, ["table_hover"], gripper=self.close_g[arm],
                      label=f"{arm} 夹起", speed=0.9, timeout=4.0)
            if not self.lift_verify(arm):
                self.log.warning("[%s] 抬起后滑脱，重试", arm)
                self._retry_table_pick(arm, timer)
                return
            self.move(arm, ["carry"], gripper=self.close_g[arm],
                      label=f"{arm} 去搬运位", speed=1.0, timeout=4.0)
        finally:
            self.release(arm, "table")

    def _retry_table_pick(self, arm: str, timer: PhaseTimer):
        grasp = self.cfg["grasp"]
        for attempt in range(1, int(grasp["retries"]) + 1):
            timer.check()
            self.move(arm, ["table_hover"], gripper=self.open_g[arm],
                      label=f"{arm} 回观察位", speed=1.0, timeout=4.0)
            q_hover, det = self.align_at_hover(arm, "table_hover")
            if det is None:
                q_hover, det = self.scan_for_cube(arm, "table_hover")
            if det is None:
                raise C.TargetMissing(f"{arm} 重试时找不到方块")
            # 每次重试略微加深
            d = self.vertical_delta[(arm, "table_hover")].copy()
            d[1] += float(grasp["retry_deepen"]) * attempt
            target = np.clip(q_hover + d, -100, 100)
            self.move(arm, [target], gripper=self.open_g[arm],
                      label=f"{arm} 重试下探 {attempt}", precision=True, timeout=5.0)
            if self.close_and_verify(arm):
                self.move(arm, ["table_hover"], gripper=self.close_g[arm],
                          label=f"{arm} 夹起", speed=0.9, timeout=4.0)
                if self.lift_verify(arm):
                    self.move(arm, ["carry"], gripper=self.close_g[arm],
                              label=f"{arm} 去搬运位", speed=1.0, timeout=4.0)
                    return
            self.release_gripper(arm)
        raise C.GraspFailed(f"{arm} 桌面抓取 {grasp['retries']} 次都失败")

    def _align_ok(self, arm: str, det: Optional[Detection]) -> bool:
        target = self.aligners[arm].target_px()
        if target is None or det is None:
            return True
        return float(np.abs(det.center - target).max()) <= float(
            self.cfg["grasp"]["align_tol_px"]) * 3.0

    def phase_place_in_box(self, arm: str):
        self.acquire(arm, "box")
        try:
            timer = PhaseTimer(float(self.cfg["safety"]["phase_timeout_s"]), f"{arm} 放入盒子")
            self.log.info("[%s] 阶段：放入盒子", arm)
            self.move(arm, ["box_approach"], gripper=self.close_g[arm],
                      label=f"{arm} 去盒子上方", speed=1.0, timeout=5.0)
            self.move(arm, ["box_hover"], gripper=self.close_g[arm],
                      label=f"{arm} 盒口", speed=0.85, timeout=4.0)
            timer.check()
            self.descend_to(arm, self.planned_q(arm, "box_hover"),
                            self.drop_delta[arm], gripper=self.close_g[arm],
                            label=f"{arm} 竖直放入")
            # 先停再松：原地张开、停一会儿让方块落稳，再抬回盒口。
            # 边抬边松会省 0.2s，但方块容易被抬起的夹爪带一下，落点不稳。
            self.release_here(arm, "box_hover", label=f"{arm} 松爪抬离")
            if not self._verify_in_box(arm):
                self._retry_place_in_box(arm)
            self.move(arm, ["box_approach", "carry"], gripper=self.open_g[arm],
                      label=f"{arm} 离开盒子", speed=1.0, timeout=5.0)
            # 左臂放完立即让出盒子；右臂已在 carry 待命，顺手预位到盒口
            if arm == "left":
                self._preposition("right", "box_approach")
        finally:
            self.release(arm, "box")

    def _verify_in_box(self, arm: str) -> bool:
        """确认方块落进盒子，并且落在标定的落点上。

        返回 False 表示“确实看到方块但偏了”，调用方据此重放一次。
        完全看不到方块时返回 True：可能是相机角度/遮挡，不能凭一次
        没检测到就判定失败（重放反而会把已经放好的方块再折腾一遍）。
        """
        roi = self.box_roi[arm]
        frame, det = self.detect_at(arm, roi=roi, force=True)
        if det is None:
            self.log.warning("[%s] 盒内 ROI 没看到方块（继续，靠下一步抓取验证）", arm)
            if frame is not None:
                save_snapshot(self.cfg, self.det, frame, det,
                              f"{arm}_box_check_fail", self.log)
            return True
        target = self.drop_px.get(arm)
        if target is None:
            self.log.info("[%s] 盒内确认到方块: %s（未标 drop_px，不判落点）", arm, det)
            return True
        target = target + self.drop_offset.get(arm, np.zeros(2))
        err = float(np.abs(det.center - target).max())
        tol = float(self.cfg["grasp"].get("drop_tol_px", 25.0))
        if err <= tol:
            self.log.info("[%s] 落点误差 %.0fpx（容差 %.0f）", arm, err, tol)
            return True
        self.log.warning("[%s] 落点偏了 %.0fpx（容差 %.0f）：方块 %s，目标 %s",
                         arm, err, tol, np.round(det.center, 0), np.round(target, 0))
        save_snapshot(self.cfg, self.det, frame, det, f"{arm}_drop_off", self.log)
        return False

    def _retry_place_in_box(self, arm: str):
        """落点偏了就重新夹起来再放一次，最多一次，避免无限套娃。"""
        if self._placing_retry.get(arm):
            return
        self._placing_retry[arm] = True
        try:
            self.log.warning("[%s] 重新夹起方块，再放一次", arm)
            self.phase_pick_from_box(arm, attempt_note="（重新放置）")
            self.phase_place_in_box(arm)
        except C.StopRequested:
            raise
        except Exception as exc:
            self.log.error("[%s] 重放失败: %s", arm, exc)
        finally:
            self._placing_retry[arm] = False

    # ------------------------------------------------------------------
    # 阶段 2/3：右臂 盒内取出 -> 放回
    # ------------------------------------------------------------------
    def _blind_pick_from_box(self, arm: str, timer: PhaseTimer) -> bool:
        """按标定位姿竖直下探抓盒内方块（视觉完全失效时的兜底）。

        实测：标定 pan 下竖直下探（深度=vertical_delta，已含 2° 高度余量）
        能稳定夹住；失败基本都发生在额外加 pan 偏移把 shoulder_pan 推到
        ±100 饱和时——关节被截断、臂不再横移，夹爪贴着方块旁边顶到盒底，
        合爪读数看着像夹住（load 瞬时偏高），一抬就滑脱。
        所以这里只做小幅、不饱和的 pan/腕转变体，深度不再加深。
        """
        lift_speed = float(self.cfg["grasp"].get("lift_speed", 0.3))
        # vertical_delta 已扣掉 box_pick_height_offset（正数=更高），直接用；
        # 再加回去等于抵消安全余量，夹爪会压到盒底/桌面。
        delta = self.vertical_delta[(arm, "box_hover")].copy()
        base_pose = self.pose(arm, "box_hover")
        # 候选顺序按实测：先原位，再小幅换腕转，最后小幅横移。
        if arm == "right":
            attempts = ((0.0, 0.0), (0.0, 15.0), (0.0, -15.0),
                        (-6.0, 0.0), (6.0, 0.0))
        else:
            attempts = ((0.0, 0.0), (0.0, 15.0), (0.0, -15.0),
                        (6.0, 0.0), (-6.0, 0.0))
        for pan, roll in attempts:
            timer.check()
            # 单次尝试约 15~20s；剩余时间不够就别开新的一轮，
            # 免得刚下探到一半就抛阶段超时（会把方块留在半路）。
            if timer.remaining < 14.0:
                self.log.warning("[%s] 盒内盲抓剩余 %.0fs，不再尝试",
                                 arm, timer.remaining)
                break
            if abs(base_pose[0] + pan) >= 99.0:
                self.log.warning("[%s] 盒内盲抓 pan %+.0f 会到 %.0f（饱和），跳过",
                                 arm, pan, base_pose[0] + pan)
                continue
            self.move(arm, ["box_hover"], gripper=self.open_g[arm],
                      label=f"{arm} 盒内观察位", speed=1.0, timeout=4.0)
            base = self.planned_q(arm, "box_hover").copy()
            if abs(roll) > 1e-6:
                q = base.copy()
                q[4] += roll
                self.move(arm, [np.clip(q, -100.0, 100.0)],
                          gripper=self.open_g[arm],
                          label=f"{arm} 盒内转腕 {roll:+.0f}", speed=0.8,
                          timeout=4.0)
                base = self.planned_q(arm, "box_hover").copy()
            d = delta.copy()
            d[0] += pan
            target = np.clip(base + d, -100.0, 100.0)
            self.log.info("[%s] 盒内盲抓：pan %+.0f° 腕转 %+.0f°，下探到 %s",
                          arm, pan, roll, C.fmt_q(target))
            # 盒内下探经常停在 3~4 的残差（方块/盒壁轻微顶住），沿用全局
            # 容差 3.0 会抛阶段错误。这里只要求到 6 以内，位置足够夹取。
            self.move(arm, [target], gripper=self.open_g[arm],
                      label=f"{arm} 盒内下探", precision=True, timeout=8.0,
                      arrive_tol=6.0)
            if self.close_and_verify(arm):
                self.move(arm, ["box_hover"], gripper=self.close_g[arm],
                          label=f"{arm} 夹出", speed=lift_speed, timeout=8.0)
                if self.lift_verify(arm):
                    self.move(arm, ["box_approach", "carry"],
                              gripper=self.close_g[arm],
                              label=f"{arm} 取出到搬运位", speed=1.0,
                              timeout=5.0)
                    return True
                self.log.warning("[%s] 盒内盲抓抬起滑脱", arm)
            self.release_gripper(arm)
            self.move(arm, ["box_hover"], gripper=self.open_g[arm],
                      label=f"{arm} 退出盒子", speed=1.0, timeout=4.0)
        return False

    def _visual_pick_from_box(self, arm: str, timer: PhaseTimer) -> bool:
        """按视觉把盒内方块夹出来（主路径，与左臂桌面抓取同一套流程）。

        观察位先用视觉确认方块在哪、必要时用 pan 把方块对到抓取中心像素，
        然后竖直下探（不做逐段对中）。盒内 lift 增量约 48°，相机跟着臂下降，
        方块纵向像素本来就会一路漂移，逐段对中会把臂往回抬，合爪位置高于
        方块。实测：标定 pan 竖直下探 + 2° 高度余量即可夹稳。
        """
        grasp = self.cfg["grasp"]
        lift_speed = float(grasp.get("lift_speed", 0.3))
        # vertical_delta 已扣掉 box_pick_height_offset（正数=更高），直接用；
        # 再加回去等于抵消安全余量，夹爪会压到盒底/桌面。
        delta = self.vertical_delta[(arm, "box_hover")].copy()
        # 重试时每轮再深一点：第一轮按抬高后的安全高度下探，如果只夹到方块
        # 顶棱（闭爪 pos 只比空夹高一点、抬起就滑），第二轮补 1.2° 让两指
        # 落到方块中部。加深量有上限，且下探本身触底会提前停。
        deepen = float(grasp.get("box_retry_deepen", 1.2))
        target_x = self.box_center_px[arm]
        attempts = max(1, int(grasp.get("box_visual_tries", 2)))
        for attempt in range(1, attempts + 1):
            timer.check()
            if timer.remaining < 12.0:
                self.log.warning("[%s] 盒内视觉抓取剩余 %.0fs，停止尝试",
                                 arm, timer.remaining)
                return False
            self.move(arm, ["box_hover"], gripper=self.open_g[arm],
                      label=f"{arm} 盒内观察位", speed=0.9, timeout=4.0)
            roi = self.box_roi[arm]
            _, det = self.detect_at(arm, roi=roi, force=True)
            if det is None:
                self.log.warning("[%s] 盒内观察位没看到方块，扫描一次", arm)
                _, det = self.scan_for_cube(arm, "box_hover", roi=roi)
            if det is None:
                self.log.warning("[%s] 盒内完全看不到方块，视觉抓取放弃", arm)
                return False
            self.log.info("[%s] 盒内视觉抓取第 %d 次：方块 %s", arm, attempt, det)
            # 只在观察位横向对准：把方块推到抓取中心像素（标定 pan 下
            # 竖直下探能夹住方块时，方块在 hover 画面里的横坐标）。
            # 用实测 pan 增益，最多两轮，每轮重新检测——一次到位就够，
            # 但盒底方块会滑动，增益也不是完全线性，量一次校一次更稳。
            # 方向：gain = d(px)/d(pan)，要让像素误差归零需 dpan = -dx/gain。
            base = self.planned_q(arm, "box_hover").copy()
            gain = self.box_px_gain[arm]
            if target_x is not None and gain:
                tx = float(np.ravel(target_x)[0])
                ty = float(np.ravel(target_x)[1]) if np.ravel(target_x).size > 1 else None
                # 纵向也要在合理范围：盒壁反光偶尔会被当成方块（实测出现过
                # y=106 的误检，比正常位置高 ~90px），拿这种检测去动 pan 会把
                # 臂带偏，不如按标定位姿直接下探。
                if ty is not None and abs(float(det.center[1]) - ty) > 70.0:
                    self.log.warning(
                        "[%s] 盒内检测 y=%.0f 偏离预期 %.0f，判为不可靠，"
                        "不做横向对准", arm, det.center[1], ty)
                    gain = None
            if target_x is not None and gain:
                for it in range(2):
                    dx = float(det.center[0]) - tx
                    # 死区 60px：盒内方块本来就会在盒底小范围滑动，
                    # 标定 pan 下竖直下探实测能夹住，只有明显偏了才动臂。
                    if abs(dx) <= 60.0:
                        break
                    dpan = float(np.clip(-dx / gain, -18.0, 18.0))
                    new_pan = float(np.clip(base[0] + dpan, -99.0, 99.0))
                    if abs(new_pan - base[0]) < 1.0:
                        # 已经贴着 pan 限位了：再发指令关节也不会动，
                        # 直接按当前位姿下探，让合爪验证去判成败。
                        self.log.warning(
                            "[%s] 盒内横向对准到 pan %.0f 已到限位，跳过",
                            arm, base[0])
                        break
                    if abs(new_pan - base[0]) < abs(dpan) - 0.5:
                        self.log.info("[%s] 盒内横向对准受 pan 限位截断："
                                      "%.1f -> %.1f", arm, base[0], new_pan)
                    base[0] = new_pan
                    self.move(arm, [base], gripper=self.open_g[arm],
                              label=f"{arm} 盒内横向对准", speed=0.8,
                              precision=True, timeout=4.0)
                    self.log.info(
                        "[%s] 盒内横向对准 %d: dx=%+.0fpx 增益 %.2fpx/° "
                        "-> pan %+.1f", arm, it + 1, dx, gain, base[0])
                    _, det2 = self.detect_at(arm, roi=roi, force=True)
                    if det2 is None:
                        self.log.warning("[%s] 对准后看不到方块，用当前位姿下探",
                                         arm)
                        break
                    det = det2
            d = delta.copy()
            if deepen > 0.0:
                d[1] += deepen * (attempt - 1)
            self.descend_tracking(
                arm, "box_hover", d, gripper=self.open_g[arm], roi=roi,
                label=f"{arm} 盒内下探", track=False, base=base)
            timer.check()
            if self.close_and_verify(arm):
                self.move(arm, ["box_hover"], gripper=self.close_g[arm],
                          label=f"{arm} 夹出", speed=lift_speed, timeout=8.0)
                if self.lift_verify(arm):
                    self.move(arm, ["box_approach", "carry"],
                              gripper=self.close_g[arm],
                              label=f"{arm} 取出到搬运位", speed=1.0,
                              timeout=5.0)
                    return True
                self.log.warning("[%s] 盒内视觉抓取抬起滑脱", arm)
            else:
                self.log.warning("[%s] 盒内视觉抓取第 %d 次没夹住", arm, attempt)
            self.release_gripper(arm)
        return False

    def phase_pick_from_box(self, arm: str, attempt_note: str = ""):
        self.acquire(arm, "box")
        try:
            timer = PhaseTimer(float(self.cfg["safety"].get(
                "box_pick_timeout_s", self.cfg["safety"]["phase_timeout_s"])),
                f"{arm} 盒内抓取")
            self.log.info("[%s] 阶段：盒内抓出 %s", arm, attempt_note)
            self.move(arm, ["carry", "box_approach"], gripper=self.open_g[arm],
                      label=f"{arm} 去盒子上方", speed=1.0, timeout=5.0)
            # 视觉优先；只有完全看不到方块（相机掉线/被挡）时才退回盲抓。
            if self._visual_pick_from_box(arm, timer):
                return
            self.log.warning("[%s] 盒内视觉抓取没成功，退回盲抓", arm)
            if self._blind_pick_from_box(arm, timer):
                return
            raise C.GraspFailed(f"{arm} 盒内抓取失败（视觉 + 盲抓）")
        finally:
            self.release(arm, "box")

    def _retry_box_pick(self, arm: str, timer: PhaseTimer):
        """落点偏了要重新夹起来时的入口，同样走视觉优先。"""
        if self._visual_pick_from_box(arm, timer):
            return
        if self._blind_pick_from_box(arm, timer):
            return
        raise C.GraspFailed(f"{arm} 盒内重新抓取失败")

    def phase_place_in_box_again(self, arm: str):
        """右臂把拿出来的方块放回盒子（不离开盒子区域）。"""
        self.acquire(arm, "box")
        try:
            timer = PhaseTimer(float(self.cfg["safety"]["phase_timeout_s"]), f"{arm} 放回盒子")
            self.log.info("[%s] 阶段：把方块放回盒子", arm)
            self.move(arm, ["box_approach"], gripper=self.close_g[arm],
                      label=f"{arm} 回盒子上方", speed=1.0, timeout=5.0)
            self.move(arm, ["box_hover"], gripper=self.close_g[arm],
                      label=f"{arm} 盒口", speed=0.85, timeout=4.0)
            timer.check()
            self.descend_to(arm, self.planned_q(arm, "box_hover"),
                            self.drop_delta[arm], gripper=self.close_g[arm],
                            label=f"{arm} 竖直放回")
            self.release_here(arm, "box_hover", label=f"{arm} 松爪抬离")
            if not self._verify_in_box(arm):
                self._retry_place_in_box(arm)
            # 右臂放回后回 home 让位：下一步左臂要进盒子
            self.move(arm, ["box_approach", "carry", "home"],
                      gripper=self.open_g[arm],
                      label=f"{arm} 放回后回 home", speed=1.0, timeout=6.0)
            self._preposition("left", "box_approach")
        finally:
            self.release(arm, "box")

    # ------------------------------------------------------------------
    # 阶段 4（可选，loop.mode == "table"）：左臂 盒内取出 -> 放回桌面
    # ------------------------------------------------------------------
    def phase_place_on_table(self, arm: str):
        self.acquire(arm, "table")
        try:
            timer = PhaseTimer(float(self.cfg["safety"]["phase_timeout_s"]), f"{arm} 放回桌面")
            self.log.info("[%s] 阶段：放回桌面", arm)
            self.move(arm, ["carry", "table_approach"], gripper=self.close_g[arm],
                      label=f"{arm} 去桌面", speed=1.0, timeout=5.0)
            self.move(arm, ["table_hover"], gripper=self.close_g[arm],
                      label=f"{arm} 桌面释放上方", speed=0.9, timeout=4.0)
            timer.check()
            self.descend_to(arm, self.planned_q(arm, "table_hover"),
                            self.drop_delta[arm + "_table"],
                            gripper=self.close_g[arm], label=f"{arm} 竖直放下")
            self.release_here(arm, "table_hover", label=f"{arm} 松爪抬起")
            frame, det = self.detect_at(arm, force=True)
            if det is not None:
                self.log.info("[%s] 桌面确认到方块: %s", arm, det)
            else:
                self.log.warning("[%s] 桌面上没看到方块（下一轮会再次确认）", arm)
                save_snapshot(self.cfg, self.det, frame, det,
                              f"{arm}_table_check_fail", self.log)
            self.move(arm, ["table_approach", "carry", "home"], gripper=self.open_g[arm],
                      label=f"{arm} 回 home", speed=1.0, timeout=6.0)
        finally:
            self.release(arm, "table")

    # ------------------------------------------------------------------
    # 一轮
    # ------------------------------------------------------------------
    def _preposition(self, arm: str, key: str):
        """让空闲臂提前就位，等真正用到时只走最后一段，减少停顿。"""
        if not bool(self.cfg["motion"].get("overlap_idle_arm", True)):
            return
        a = self.arms[arm]
        if a.busy():
            return
        try:
            a.start([self.pose(arm, key)], gripper=self.open_g[arm],
                    label=f"{arm} 预位 {key}", speed_scale=1.0)
        except Exception as exc:
            self.log.debug("[%s] 预位失败: %s", arm, exc)

    def run_cycle(self, index: int) -> None:
        t0 = time.perf_counter()
        self.log.info("=" * 66)
        self.log.info("第 %d 轮开始", index)
        self._check_temp()

        mode = str(self.cfg["loop"].get("mode", "in_box"))
        left_holds = bool(self.state["left"].holding)

        # 首轮（或上一轮把方块放回桌面了）左臂需要先从桌面抓起
        if not left_holds:
            self._preposition("right", "carry")
            self.phase_pick_from_table("left")
        else:
            self.log.info("[left] 上一轮已取出方块，本轮直接放入盒子")
            self._preposition("right", "carry")

        # 1. 左臂：方块放入盒内
        self.phase_place_in_box("left")

        # 2. 右臂：从盒内把方块夹出来
        self.phase_pick_from_box("right", attempt_note="（取出）")

        # 3. 右臂：把拿出来的方块放回盒中
        self.phase_place_in_box_again("right")

        # 4. 左臂：把盒内的方块夹出来
        self.phase_pick_from_box("left", attempt_note="（取出）")

        if mode == "table":
            # 可选：每轮结束放回桌面，下一轮再从桌面抓起（视觉更稳，稍慢）
            self.phase_place_on_table("left")
        else:
            self.log.info("[left] 方块留在左爪，下一轮循环从“放入盒子”继续")

        dt = time.perf_counter() - t0
        self.log.info("第 %d 轮完成，用时 %.1fs", index, dt)
        if dt > float(self.cfg["safety"]["cycle_timeout_s"]):
            self.log.warning("本轮 %.1fs 超过预估 cycle_timeout_s=%.1fs",
                             dt, float(self.cfg["safety"]["cycle_timeout_s"]))

    # ------------------------------------------------------------------
    # 恢复
    # ------------------------------------------------------------------
    def recover(self):
        self.log.warning("执行恢复流程")
        for a in ("left", "right"):
            try:
                self.release_gripper(a)
            except Exception:
                pass
            self.zone_owner["box"] = None
            self.zone_owner["table"] = None
        # 相机看门狗
        for a, cam in self.cams.items():
            if cam.age > 3.0:
                self.log.warning("[%s] 相机 %.1fs 无新帧，重连", a, cam.age)
                cam.reconnect()
        self._sleep(float(self.cfg["safety"]["recover_pause_s"]))
        # 先把两臂送回 home，再判断方块在哪
        for a in ("left", "right"):
            try:
                self.arms[a].start([self.P[a]["home"]], gripper=self.open_g[a],
                                   label=f"{a} 恢复回 home", speed_scale=0.8)
            except Exception:
                pass
        for a in ("left", "right"):
            try:
                self.arms[a].wait(timeout=5.0)
            except Exception:
                pass
        # 盒子区域优先：如果盒里有方块而流程本该空着，尽量让左臂取回桌面
        try:
            self._drain_box_best_effort()
        except C.StopRequested:
            raise
        except Exception as exc:
            self.log.warning("恢复时处理盒子失败: %s", exc)
        self._measure_air_closed()

    def _drain_box_best_effort(self):
        """恢复：如果盒子里有方块，用左臂取出来放回桌面。"""
        arm = "left"
        frames = {}
        for a in ("left", "right"):
            frame, det = self.detect_at(a, roi=self.box_roi[a], force=True)
            frames[a] = (frame, det)
        target_arm = next((a for a in ("left", "right") if frames[a][1] is not None), None)
        if target_arm is None:
            self.log.info("恢复：盒内没看到方块，跳过取回")
            return
        self.log.warning("恢复：%s 相机在盒内看到方块，尝试取回桌面", target_arm)
        try:
            self.phase_pick_from_box(target_arm, attempt_note="（恢复取回）")
            self.phase_place_on_table(target_arm)
        except Exception as exc:
            self.log.error("恢复取回失败: %s", exc)

    def shutdown(self):
        """停机：夹着方块的臂先放到安全位置再松爪，避免半空摔落。

        正常停止时（--cycles 到点 / --seconds 到点）左臂通常正夹着方块
        （in_box 模式）。直接开爪会让方块从半空掉到桌面，落点不可预测，
        下一轮就得靠视觉重找。Ctrl+C 是急停，保持原有的“先开爪再下电”。
        """
        safe_stop = not self.stop_event.is_set()
        for a in ("left", "right"):
            if not safe_stop or not self.state[a].holding:
                continue
            try:
                self.log.info("[%s] 停机时仍夹着方块，先放回%s", a,
                              "盒子" if a == "left" else "桌面")
                if a == "left":
                    self.phase_place_in_box(a)
                else:
                    self.phase_place_on_table(a)
            except C.StopRequested:
                self.log.info("[%s] 停机放置被停止请求打断", a)
            except Exception as exc:
                self.log.warning("[%s] 停机放置失败: %s", a, exc)
        for a in ("left", "right"):
            try:
                self.arms[a].set_gripper(self.open_g[a], settle=0.05)
            except Exception:
                pass
        try:
            self.dual.disable_torque()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def prepare(self):
        self._measure_air_closed()
        for a in ("left", "right"):
            self.log.info("[%s] 位姿: %s", a,
                          ", ".join(f"{k}={C.fmt_q(v)}" for k, v in self.P[a].items()))
            self.log.info("[%s] 竖直增量: table=%s box=%s", a,
                          C.fmt_q(self.vertical_delta[(a, "table_hover")]),
                          C.fmt_q(self.vertical_delta[(a, "box_hover")]))
            self.log.info("[%s] 夹爪: open=%.1f closed=%.1f", a,
                          self.open_g[a], self.close_g[a])
            if self.box_roi[a] is None:
                self.log.warning("[%s] 未标定 box_roi，盒内检测会用全画面", a)
        # 两臂回 home 待命
        for a in ("left", "right"):
            self.arms[a].start([self.P[a]["home"]], gripper=self.open_g[a],
                               label=f"{a} 待命", speed_scale=0.8)
        for a in ("left", "right"):
            self.arms[a].wait(timeout=6.0)


# ---------------------------------------------------------------------------
# 看门狗 / 监督
# ---------------------------------------------------------------------------

class Supervisor:
    def __init__(self, cfg: dict, logger):
        self.cfg = cfg
        self.log = logger
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.csv = CycleCSV(C.LOG_DIR / "cycles_2arm.csv")
        self.dual: Optional[DualArm] = None
        self.cams: dict[str, CameraStream] = {}
        self.detector: Optional[Detector] = None
        self.runner: Optional[LoopRunner] = None
        self._watchdog: Optional[threading.Thread] = None
        # 看门狗用它判断主循环是不是卡死了（无人值守时唯一的兜底）
        self._last_progress = time.perf_counter()

    def _signals(self):
        def handler(signum, frame):
            self.log.warning("收到信号 %s，准备停止", signum)
            self.stop_event.set()
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _watchdog_loop(self):
        err_count = 0
        while not self.stop_event.wait(1.0):
            try:
                if self.runner is None:
                    continue
                for a, cam in self.cams.items():
                    if cam.age > float(self.cfg["safety"].get("camera_stale_s", 3.0)):
                        self.log.warning("[%s] 相机 %.1fs 无新帧，重连", a, cam.age)
                        cam.reconnect()
                streak = self.dual.error_streak() if self.dual else 0
                if streak >= int(self.cfg["safety"]["reconnect_after_errors"]):
                    err_count += 1
                    self.log.error("总线错误累积 %d，尝试重连 (%d/2)", streak, err_count)
                    try:
                        self.dual.reconnect()
                    except Exception as exc:
                        self.log.error("重连失败: %s", exc)
                    if self.dual is not None:
                        self.dual.arms["left"].bus._errors = 0
                        self.dual.arms["right"].bus._errors = 0
                    if err_count >= 2:
                        self.log.error("多次总线错误，暂停 5s")
                        time.sleep(5.0)
                        err_count = 0
                # 主循环卡死检测：正常的某一轮最多 cycle_timeout_s 量级，
                # 超过 stall_restart_s 还没有任何进展就判定卡死。直接退出，
                # 由 systemd Restart=always 拉起来——无人值守时这是唯一
                # 能把“卡在半路”救回来的办法。
                stall = float(self.cfg["safety"].get("stall_restart_s", 180.0))
                if stall > 0 and time.perf_counter() - self._last_progress > stall:
                    self.log.error("主循环 %.0fs 没有任何进展，退出让 systemd 重启",
                                   stall)
                    try:
                        if self.dual is not None:
                            self.dual.disable_torque()
                    except Exception:
                        pass
                    os._exit(3)
            except Exception as exc:
                self.log.debug("看门狗异常: %s", exc)

    def _setup(self):
        for a in ("left", "right"):
            missing = C.missing_poses(self.cfg, a)
            if missing:
                raise SystemExit(
                    f"[{a}] 缺少 {len(missing)} 个位姿: {missing}\n"
                    f"请先运行: python calibrate2arm.py --arm {a}")
            ac = C.arm_cfg(self.cfg, a)
            if ac.get("gripper_open") is None or ac.get("gripper_closed") is None:
                raise SystemExit(f"[{a}] 夹爪开合未标定: "
                                 f"python calibrate2arm.py --arm {a}")
            bad = C.pose_saturation_report(self.cfg, a)
            if bad:
                raise SystemExit(
                    f"[{a}] 这些位姿的关节读数顶到 ±100，已被标定范围截断:\n"
                    + "\n".join(bad)
                    + "\n截断后不同高度会变成同一个值，下探会撞限位。\n"
                    f"请重标这些点: python calibrate2arm.py --arm {a}")
            if (C.arm_cfg(self.cfg, a).get("gains") or {}).get("box_roi") is None:
                # 没有 box_roi 时盒内检测会退化成全画面，可能把桌面上的
                # 方块当成盒内的抓走。宁可不启动。
                raise SystemExit(
                    f"[{a}] 没有 box_roi（盒内检测范围），盒内抓取会退化成全画面检测。\n"
                    f"请重标 box_hover/box_drop（盒里放一个方块）: "
                    f"python calibrate2arm.py --arm {a}")
        self.dual = DualArm(self.cfg, self.log, self.stop_event)
        self.dual.connect()
        # 每个点位是在哪套电机标定下录的？逐点位比对，不匹配就不能跑：
        # 归一化坐标系变了，那些点位会指向错误角度。中途退出标定会导致
        # 一条臂上新旧点位混在一起，所以必须逐点查。
        for a in ("left", "right"):
            bus = getattr(self.dual.arms[a].bus, "bus", None)
            cal = getattr(bus, "calibration", None) if bus is not None else None
            if not cal:
                continue
            differs, unknown = C.pose_calib_mismatch(self.cfg, a, cal)
            if differs:
                raise SystemExit(
                    f"[{a}] 这些点位录制的电机标定与当前硬件不一致: {differs}\n"
                    f"这段时间重标过电机、换过臂，或者标定中途退出过，\n"
                    f"导致一部分点位是新坐标、一部分是旧坐标。\n"
                    f"请把该臂 10 个点位完整重标（不要中途退出）:\n"
                    f"  python calibrate2arm.py --arm {a}")
            if unknown:
                raise SystemExit(
                    f"[{a}] 这些点位没有标定指纹，无法确认属于当前电机标定: {unknown}\n"
                    f"它们是加入指纹机制之前录的，或者标定中途退出过。\n"
                    f"为了安全，拒绝对这种情况启动——旧坐标可能让臂撞限位。\n"
                    f"请完整重标该臂 10 个点位: python calibrate2arm.py --arm {a}")
            # 接近边界但不等于截断：只警告，不拦启动。
            tight = C.tight_pose_report(self.cfg, a, cal)
            if tight:
                self.log.warning("[%s] 这些点位离标定边界不到 5，插值或视觉微调时\n"
                                 "可能被截断到 ±100（那一段会发指令但不动）:\n%s",
                                 a, "\n".join(tight))
        for a in ("left", "right"):
            dev = cam_device(self.cfg, a)
            cam = CameraStream(dev, logger=self.log, name=a,
                               width=self.cfg["devices"]["camera_width"],
                               height=self.cfg["devices"]["camera_height"],
                               fps=self.cfg["devices"]["camera_fps"])
            cam.start()
            self.cams[a] = cam
        self.detector = Detector(self.cfg, self.log)
        self.runner = LoopRunner(self.cfg, self.dual, self.cams, self.detector,
                                 self.stop_event, self.log, self.csv)
        # 启动前置：在 measure 位测空夹基线（抓取判定要用），再把两臂摆到
        # home 待命。以前这个函数只在离线自测里被调用，真实运行从没执行过，
        # 首轮的空夹阈值是空的，抓取判定退化成只剩“位置条”。
        self.runner.prepare()

    def _teardown(self):
        if self.runner is not None:
            try:
                self.runner.shutdown()
            except Exception:
                pass
        for cam in self.cams.values():
            try:
                cam.stop()
            except Exception:
                pass
        if self.dual is not None:
            try:
                self.dual.stop_workers()
            except Exception:
                pass
            try:
                self.dual.disconnect()
            except Exception:
                pass
        if self.detector is not None:
            self.detector.close()

    def run(self, max_cycles: int = 0, max_seconds: float = 0.0,
            start_from_box: bool = False):
        self._signals()
        attempt = 0
        while not self.stop_event.is_set():
            attempt += 1
            try:
                self._setup()
                break
            except SystemExit:
                raise
            except Exception as exc:
                self.log.exception("初始化失败 (%d): %s", attempt, exc)
                self._teardown()
                if self.stop_event.wait(5.0):
                    return
        if self.stop_event.is_set():
            self._teardown()
            return

        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True,
                                          name="watchdog")
        self._watchdog.start()

        # 方块一开始就在盒里时，先走一小段“右臂取出 -> 放回 -> 左臂取出”，
        # 让左臂拿住方块，之后主循环就能从“放入盒子”开始无缝往复。
        if start_from_box and not self.stop_event.is_set():
            self.log.info("起始状态：方块在盒内，先执行起始循环")
            try:
                self.runner.phase_pick_from_box("right", attempt_note="（起始取出）")
                self.runner.phase_place_in_box_again("right")
                self.runner.phase_pick_from_box("left", attempt_note="（起始取出）")
                self.log.info("起始循环完成，左臂已取出方块，进入主循环")
            except C.StopRequested:
                self._teardown()
                return
            except Exception as exc:
                self.log.error("起始循环失败: %s", exc)
                self.log.debug(traceback.format_exc())

        t0 = time.perf_counter()
        cycle = 0
        consecutive = 0
        try:
            while not self.stop_event.is_set():
                if max_seconds and time.perf_counter() - t0 >= max_seconds:
                    self.log.info("达到 --seconds=%.0f，正常停止", max_seconds)
                    break
                if max_cycles and cycle >= max_cycles:
                    break
                cycle += 1
                c0 = time.perf_counter()
                phase = "cycle"
                err = ""
                ok = False
                try:
                    self.runner.run_cycle(cycle)
                    ok = True
                    self.runner.success += 1
                    consecutive = 0
                except C.StopRequested:
                    self.log.warning("收到停止请求，退出循环")
                    break
                except Exception as exc:
                    consecutive += 1
                    self.runner.failures += 1
                    err = f"{type(exc).__name__}: {exc}"
                    phase = "recover"
                    self.log.error("第 %d 轮失败: %s", cycle, err)
                    self.log.debug(traceback.format_exc())
                    try:
                        self.runner.recover()
                    except C.StopRequested:
                        break
                    except Exception as exc2:
                        self.log.error("恢复失败: %s", exc2)
                    if consecutive >= int(self.cfg["safety"]["max_consecutive_failures"]):
                        self.log.error("连续失败 %d 次，硬恢复（卸力 5s 后重连）",
                                       consecutive)
                        try:
                            self.dual.disable_torque()
                            self.stop_event.wait(5.0)
                            self.dual.connect()
                            self.runner._measure_air_closed()
                        except Exception as exc3:
                            self.log.error("硬恢复失败: %s", exc3)
                        consecutive = 0
                dt = time.perf_counter() - c0
                self._last_progress = time.perf_counter()
                self.csv.write(cycle, ok, dt, phase, err)
                self.log.info("统计: 成功 %d / 失败 %d / 连续失败 %d，本轮 %.1fs",
                              self.runner.success, self.runner.failures, consecutive, dt)
                snap_n = int(self.cfg["loop"].get("snapshot_every_n_cycles", 0) or 0)
                if snap_n and cycle % snap_n == 0:
                    self._snapshot(cycle)
                pause = float(self.cfg["loop"]["cycle_pause_s"])
                if pause > 0:
                    self._sleep_checked(pause)
        finally:
            self.log.info("退出清理...")
            self._teardown()
            total = time.perf_counter() - t0
            self.log.info("运行 %.1fs，共 %d 轮，成功 %d，失败 %d",
                          total, cycle, self.runner.success if self.runner else 0,
                          self.runner.failures if self.runner else 0)

    def _sleep_checked(self, seconds: float):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end and not self.stop_event.is_set():
            time.sleep(0.02)

    def _snapshot(self, cycle: int):
        for a, cam in self.cams.items():
            frame = cam.fresh(max_age=1.0)
            if frame is None:
                continue
            det = self.detector.detect(frame, cache_key=a, force_yolo=True)
            save_snapshot(self.cfg, self.detector, frame, det,
                          f"cycle{cycle:05d}_{a}", self.log)
        self._prune_snapshots()

    def _prune_snapshots(self):
        """只留最近 N 张周期快照，否则 24 小时会堆出几万张图。"""
        keep = int(self.cfg["loop"].get("snapshot_keep", 200) or 0)
        if keep <= 0:
            return
        try:
            d = C.CONFIG_PATH.parent / str(self.cfg["loop"].get("snapshot_dir", "snapshots"))
            files = sorted(d.glob("cycle*.jpg"))
            for f in files[:-keep]:
                f.unlink(missing_ok=True)
        except Exception as exc:
            self.log.debug("清理快照失败: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_self_test(cfg: dict, logger) -> int:
    print("=" * 70)
    print("双臂自检（只读，不动电机）")
    print("=" * 70)
    problems = 0
    print("\n[1] 串口")
    counts = probe_port_counts()
    for p, n in counts.items():
        print(f"  {p}: {n} 个电机")
    if len([p for p, n in counts.items() if n >= 6]) < 2:
        print("  ! 需要两条臂各至少 6 个电机")
        problems += 1
    try:
        ports = resolve_arm_ports(cfg, logger)
        print(f"  left={ports['left']}  right={ports['right']}")
    except Exception as exc:
        print(f"  ! {exc}")
        problems += 1

    print("\n[2] 位姿")
    for a in ("left", "right"):
        miss = C.missing_poses(cfg, a)
        if miss:
            print(f"  {a}: 缺 {miss}")
            problems += 1
        else:
            print(f"  {a}: 10 个位姿完整")
        ac = C.arm_cfg(cfg, a)
        if ac.get("gripper_open") is None or ac.get("gripper_closed") is None:
            print(f"  {a}: 夹爪未标定")
            problems += 1
        sat = C.pose_saturation_report(cfg, a)
        if sat:
            print(f"  {a}: 这些位姿读数被截断到 ±100，必须重标")
            for line in sat:
                print(line)
            problems += 1
        dup = C.duplicate_pose_report(cfg, a)
        if dup:
            for line in dup:
                print(line.strip())
            problems += 1
        for line in C.pose_delta_report(cfg, a):
            print(line)

    print("\n[3] 相机 + 检测")
    det = Detector(cfg, logger)
    for a in ("left", "right"):
        cam = CameraStream(cam_device(cfg, a), logger=logger, name=a)
        cam.start()
        time.sleep(0.5)
        frame = None
        for _ in range(10):
            frame = cam.fresh(max_age=1.5)
            if frame is not None:
                break
            time.sleep(0.1)
        if frame is None:
            print(f"  {a}: 没有画面")
            problems += 1
        else:
            d = det.detect(frame, cache_key=a, force_yolo=True)
            print(f"  {a}: {frame.shape[1]}x{frame.shape[0]} fps≈{cam.fps_est:.1f} "
                  f"检测={'无' if d is None else d}")
            save_snapshot(cfg, det, frame, d, f"selftest_{a}", logger)
        cam.stop()
    print(f"  YOLO: {'已加载' if det.model is not None else '未加载（HSV 兜底）'}")
    det.close()

    print()
    if problems:
        print(f"发现 {problems} 个问题。")
        return 1
    print("自检通过。下一步：python dualarm_loop.py run --cycles 1")
    return 0


def cmd_preview(cfg: dict, logger, arm: str, seconds: float, hover_key: Optional[str],
                display: bool):
    import cv2

    stop_event = threading.Event()
    cam = CameraStream(cam_device(cfg, arm), logger=logger, name=arm)
    cam.start()
    det = Detector(cfg, logger)
    dual = None
    if hover_key:
        dual = DualArm(cfg, logger, stop_event)
        dual.connect()
        a = dual.arms[arm]
        a.start([C.get_pose(cfg, arm, hover_key)], gripper=float(
            C.arm_cfg(cfg, arm)["gripper_open"]), label=f"{arm} 预览位", speed_scale=0.8)
        a.wait(timeout=6.0)
    t0 = time.time()
    last = 0.0
    try:
        while time.time() - t0 < seconds:
            frame = cam.fresh(max_age=1.0)
            if frame is None:
                time.sleep(0.05)
                continue
            d = det.detect(frame, cache_key=arm, force_yolo=True)
            if display:
                vis = det.annotate(frame.copy(), d)
                cv2.imshow(f"preview {arm}", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif time.time() - last > 0.5:
                last = time.time()
                print(f"[{time.time() - t0:5.1f}s] {'无目标' if d is None else d}")
    finally:
        cam.stop()
        if dual is not None:
            dual.stop_workers()
            dual.disconnect()
        det.close()
        if display:
            cv2.destroyAllWindows()
    return 0


def cmd_home(cfg: dict, logger) -> int:
    stop_event = threading.Event()
    dual = DualArm(cfg, logger, stop_event)
    dual.connect()
    try:
        for a in ("left", "right"):
            dual.arms[a].start([C.get_pose(cfg, a, "home")],
                               gripper=float(C.arm_cfg(cfg, a)["gripper_open"]),
                               label=f"{a} home", speed_scale=0.7)
        for a in ("left", "right"):
            dual.arms[a].wait(timeout=8.0)
        time.sleep(0.2)
        for a in ("left", "right"):
            dual.arms[a].set_gripper(float(C.arm_cfg(cfg, a)["gripper_open"]), settle=0.1)
        print("两臂已回 home。")
    finally:
        dual.stop_workers()
        try:
            dual.disable_torque()
        except Exception:
            pass
        dual.disconnect()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="双臂方块循环夹放（2arm）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("self-test", help="只读自检")

    p_prev = sub.add_parser("preview", help="相机/检测预览")
    p_prev.add_argument("--arm", choices=["left", "right"], default="left")
    p_prev.add_argument("--seconds", type=float, default=10.0)
    p_prev.add_argument("--pose", default=None,
                        help="先移动到这个位姿名，例如 table_hover（会动电机）")
    p_prev.add_argument("--display", action="store_true")

    p_run = sub.add_parser("run", help="运行循环")
    p_run.add_argument("--cycles", type=int, default=0, help="0 = 无限")
    p_run.add_argument("--seconds", type=float, default=0.0, help="0 = 不限时")
    p_run.add_argument("--start", choices=["table", "box"], default="table",
                       help="起始时方块在哪：table=桌面（默认），box=盒内")

    sub.add_parser("home", help="两臂回 home")

    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="s2b2")

    if args.cmd == "self-test":
        return cmd_self_test(cfg, logger)
    if args.cmd == "preview":
        return cmd_preview(cfg, logger, args.arm, args.seconds, args.pose, args.display)
    if args.cmd == "home":
        return cmd_home(cfg, logger)
    if args.cmd == "run":
        sup = Supervisor(cfg, logger)
        sup.run(max_cycles=args.cycles, max_seconds=args.seconds,
                start_from_box=(args.start == "box"))
        return 0
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)
