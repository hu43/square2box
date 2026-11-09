#!/usr/bin/env python
"""dualarm_cycle.py — 双臂 YOLO 视觉伺服循环夹放（无人值守 / 24h）。

动作流程（与需求逐条对应）：
    首轮引导：左臂从桌面视觉伺服抓绿色方块 -> 放入盒子
    稳态循环：左臂放入盒子 -> 右臂取出 -> 右臂放回 -> 左臂取出 -> 重复

和旧版 dualarm_loop.py 的区别：
    * 盒内抓取不再“hover 对一次 + 开环下探”，而是 pan/elbow 分段闭环跟踪；
    * 合爪前必须有新鲜、稳定的视觉检测（拒绝盲抓）；
    * 放盒后做盒内 ROI 复检，失败会重新夹起再放；
    * 每次抓取成功会学习 hover/pick 像素目标和雅可比（servo_state.json）；
    * 相机按 USB 口多节点轮换，掉帧自动切换；失败后重新定位方块再续跑。

用法（conda activate hupy3.12）：
    python dualarm_cycle.py check                    # 只读体检
    python dualarm_cycle.py home                     # 两臂回 home
    python dualarm_cycle.py run --cycles 1           # 先跑一轮，人守着
    python dualarm_cycle.py run --seconds 300        # 5 分钟验收
    python dualarm_cycle.py run                      # 无限循环（24/7）
    python dualarm_cycle.py reset-servo              # 清空学习值重新引导

停止：Ctrl+C（先开爪、把夹着的方块放回盒子，再下电）。
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
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
    box_roi_from_cfg,
    save_snapshot,
)
from servo import PAN, ELBOW, PixelServo, ServoState, ZONES


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def cam_candidates(cfg: dict, arm: str) -> list[str]:
    """该臂相机的全部候选节点：USB 物理口下的所有 /dev/videoN + 配置兜底。"""
    d = cfg["devices"]
    out: list[str] = []
    usb = d.get(f"{arm}_camera_usb")
    if usb:
        out.extend(C.find_videos_by_usb(str(usb)))
    for cand in [d.get(f"{arm}_camera")] + list(d.get(f"{arm}_camera_fallbacks") or []):
        if cand:
            out.append(str(cand))
    dedup: list[str] = []
    for x in out:
        if x not in dedup:
            dedup.append(x)
    return dedup or [f"/dev/{arm}_wrist"]


def apply_smooth(cfg: dict) -> None:
    """把 smooth 段的参数覆盖到 motion/grasp（只在内存里，不改 config.json）。"""
    sm = cfg.get("smooth") or {}
    if not bool(sm.get("apply", True)):
        return
    m = cfg["motion"]
    g = cfg["grasp"]
    m["settle_s"] = float(sm.get("settle_s", m.get("settle_s", 0.02)))
    m["arrive_tol"] = float(sm.get("arrive_tol", m.get("arrive_tol", 4.0)))
    m["arrive_timeout_s"] = float(sm.get("arrive_timeout_s", 0.25))
    m["release_dwell_s"] = float(sm.get("release_dwell_s", 0.20))
    m["gripper_open_settle_s"] = float(sm.get("gripper_open_settle_s", 0.20))
    m["gripper_close_settle_s"] = float(sm.get("gripper_close_settle_s", 0.28))
    g["lift_speed"] = float(sm.get("lift_speed", g.get("lift_speed", 0.5)))
    g["grip_hold_s"] = float(sm.get("grip_hold_s", g.get("grip_hold_s", 0.25)))
    g["load_settle_s"] = float(sm.get("load_settle_s", 0.20))
    cfg["loop"]["cycle_pause_s"] = float(sm.get("cycle_pause_s", 0.0))


class PhaseTimer:
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
    HEADER = ["ts", "cycle", "success", "duration_s", "phase", "error",
              "state_before", "state_after", "note"]

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, cycle: int, ok: bool, duration: float, phase: str, err: str,
              state_before: str = "", state_after: str = "", note: str = ""):
        new = not self.path.exists()
        try:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(self.HEADER)
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), cycle, int(ok),
                            round(duration, 2), phase, err[:300],
                            state_before, state_after, note[:300]])
        except Exception as exc:
            print(f"CSV 写入失败: {exc}")


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

@dataclass
class ArmState:
    air_closed: Optional[float] = None
    load_available: bool = False
    load_thresh: float = 0.0
    holding: bool = False


class CycleRunner:
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
        self.servo_state = ServoState(
            C.HERE / str(cfg["servo"].get("state_file", "servo_state.json")))
        self.servo = {a: PixelServo(cfg, a, logger, self.servo_state)
                      for a in ("left", "right")}

        self.P = {a: {k: C.get_pose(cfg, a, k) for k in C.POSE_KEYS}
                  for a in ("left", "right")}
        self.open_g = {a: float(C.arm_cfg(cfg, a)["gripper_open"]) for a in ("left", "right")}
        self.close_g = {a: float(C.arm_cfg(cfg, a)["gripper_closed"]) for a in ("left", "right")}
        self._grip = dict(self.open_g)
        self.table_offset = {a: float(C.arm_cfg(cfg, a).get("table_pick_height_offset") or 0.0)
                             for a in ("left", "right")}
        self.box_offset = {a: float(C.arm_cfg(cfg, a).get("box_pick_height_offset") or 0.0)
                           for a in ("left", "right")}
        self.box_roi = {a: box_roi_from_cfg(cfg, a) for a in ("left", "right")}

        self.zone_owner: dict[str, Optional[str]] = {"table": None, "box": None}
        self._zone_lock = threading.Lock()
        self.speed_factor = 1.0

        # hover -> pick 的竖直增量
        # 注意：左臂桌面 pick 与 hover 的 pan 差 6.8°、右臂 9.8°（旧标定的
        # 实际情况），把 pan 强制归零会横向偏 3cm 左右。手标的 pick 位才是
        # 人确认过“两指包住方块”的位置，所以这里保留完整增量；下探过程中
        # 的视觉跟踪会修正残余偏差。
        self.vertical_delta = {}
        for a in ("left", "right"):
            for zone, (hk, pk) in ZONES.items():
                vd = (self.P[a][pk] - self.P[a][hk]).copy()
                off = self.box_offset[a] if zone == "box" else self.table_offset[a]
                if abs(off) > 1e-6:
                    vd[1] -= off          # lift 增大 = 更低，抬高 off 要减
                self.vertical_delta[(a, zone)] = vd

        # 盒内释放增量：同样保留完整标定差，落点就是人标定的 box_drop
        self.drop_delta = {}
        for a in ("left", "right"):
            dd = (self.P[a]["box_drop"] - self.P[a]["box_hover"]).copy()
            dd[1] -= float(self.box_offset[a])
            dd[1] -= float(self.cfg["grasp"].get("box_drop_extra_lift", 0.0))
            self.drop_delta[a] = dd

        self.gate_fails = {(a, z): 0 for a in ("left", "right") for z in ZONES}
        self.hover_fails = {(a, z): 0 for a in ("left", "right") for z in ZONES}
        self.cube_state = "unknown"
        self.stats = {"servo_iters": 0, "gate_fails": 0, "pick_retries": 0}
        self.success = 0
        self.failures = 0
        self._recover_count = 0

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------
    def _check_stop(self):
        if self.stop_event.is_set():
            raise C.StopRequested()

    def _check_temp(self):
        try:
            t = self.dual.max_temp()
        except Exception:
            return
        safety = self.cfg["safety"]
        warn = float(safety["temp_warn_c"])
        slow = float(safety.get("temp_slow_c", 68.0))
        stop = float(safety["temp_stop_c"])
        if t >= stop:
            self.log.error("电机温度 %.0f°C ≥ 停机阈值 %.0f°C，卸力冷却 %.0fs",
                           t, stop, float(safety["temp_wait_s"]))
            self.dual.disable_torque()
            self._sleep(float(safety["temp_wait_s"]))
            self.dual.enable_torque()
            self.speed_factor = 1.0
            self._recover_after_pause()
        elif t >= slow:
            self.speed_factor = float(self.cfg["motion"].get("temp_slow_scale", 0.7))
            self.log.warning("电机温度 %.0f°C，降速到 %.0f%%", t, self.speed_factor * 100)
        else:
            self.speed_factor = 1.0
            if t >= warn:
                self.log.warning("电机温度偏高 %.0f°C", t)

    def _sleep(self, seconds: float):
        end = time.perf_counter() + max(0.0, seconds)
        while time.perf_counter() < end:
            self._check_stop()
            time.sleep(min(0.1, max(0.0, end - time.perf_counter())))

    def _recover_after_pause(self):
        for a in ("left", "right"):
            try:
                self.arms[a].start([self.P[a]["home"]], gripper=self.open_g[a],
                                   label=f"{a} 回 home", speed_scale=0.7)
            except Exception:
                pass
        for a in ("left", "right"):
            try:
                self.arms[a].wait(timeout=6.0)
            except Exception:
                pass
        self._measure_air_closed()

    # ---- 区域互锁 ----
    def acquire(self, arm: str, zone: str):
        with self._zone_lock:
            owner = self.zone_owner.get(zone)
            if owner is not None and owner != arm:
                raise C.PhaseError(f"{zone} 区域被 {owner} 占用")
            self.zone_owner[zone] = arm

    def release(self, arm: str, zone: str):
        with self._zone_lock:
            if self.zone_owner.get(zone) == arm:
                self.zone_owner[zone] = None

    # ---- 位姿 / 运动 ----
    def pose(self, arm: str, key: str) -> np.ndarray:
        return self.P[arm][key]

    def planned_q(self, arm: str, fallback_key: str = "home") -> np.ndarray:
        a = self.arms[arm]
        q = getattr(a, "_planned_q", None)
        if q is None:
            q = getattr(a, "hold_q", None)
        if q is None:
            q = self.pose(arm, fallback_key)
        return np.asarray(q, dtype=float)

    def move_pose(self, arm: str, keys, *, gripper=None, label="", speed: float = 1.0,
                  timeout=None, tolerate: bool = False, precision: bool = False,
                  arrive_tol=None) -> bool:
        wps = [self.pose(arm, k) if isinstance(k, str) else np.asarray(k, float)
               for k in keys]
        a = self.arms[arm]
        ref = getattr(a, "_planned_q", None)
        if ref is None:
            ref = getattr(a, "hold_q", None)
        if ref is not None and len(wps) > 1:
            start_k = 0
            for i in range(len(wps)):
                if float(np.abs(wps[i] - ref).max()) <= 1.5:
                    start_k = i
            if start_k:
                wps = wps[start_k:]
        tmo = timeout if timeout is not None else max(3.0, 0.35 * len(wps) + 3.0)
        gcmd = gripper if gripper is not None else self._grip[arm]
        a.start(wps, gripper=gcmd, label=label or ",".join(map(str, keys)),
                speed_scale=speed * self.speed_factor, precision=precision)
        ok = a.wait(timeout=tmo, tol=arrive_tol)
        self._grip[arm] = float(gcmd)
        if not ok and not tolerate:
            self.log.warning("[%s] %s 未在 %.1fs 内到位，继续", arm, label, tmo)
        self._check_stop()
        return ok

    def move_q(self, arm: str, q, *, gripper=None, label="", speed: float = 1.0,
               precision: bool = False, tolerate: bool = False,
               timeout=None, arrive_tol=None) -> bool:
        return self.move_pose(arm, [np.asarray(q, dtype=float)], gripper=gripper,
                              label=label, speed=speed, timeout=timeout,
                              tolerate=tolerate, precision=precision,
                              arrive_tol=arrive_tol)

    def _preposition(self, arm: str, key: str):
        if not bool(self.cfg["smooth"].get("preposition", True)):
            return
        a = self.arms[arm]
        if a.busy() or self.state[arm].holding:
            return
        try:
            a.start([self.pose(arm, key)], gripper=self.open_g[arm],
                    label=f"{arm} 预位 {key}", speed_scale=1.0)
        except Exception as exc:
            self.log.debug("[%s] 预位失败: %s", arm, exc)

    # ---- 视觉 ----
    def detect(self, arm: str, zone: Optional[str] = None, *, force: bool = False):
        """返回 Detection | None。zone='box' 时套 box_roi；None 表示全画面。"""
        cam = self.cams[arm]
        frame = cam.fresh(max_age=0.9)
        if frame is None:
            try:
                frame = cam.read(timeout=1.0)
            except CameraError as exc:
                self.log.warning("[%s] 相机异常: %s", arm, exc)
                return None
        roi = self.box_roi[arm] if zone == "box" else None
        return self.det.detect(frame, roi=roi, cache_key=f"{arm}:{zone or 'full'}",
                               force_yolo=force)

    def detect_stable(self, arm: str, zone: Optional[str] = None, *,
                      force: bool = False, n: int = 2,
                      max_age: float = 1.2) -> Optional[Detection]:
        """取 n 帧检测，位置一致才返回中位数结果（抑制反光跳变）。"""
        samples: list[Detection] = []
        deadline = time.perf_counter() + max(0.35, 0.09 * n + 0.2)
        while len(samples) < max(1, n) and time.perf_counter() < deadline:
            d = self.detect(arm, zone, force=force)
            if d is not None:
                samples.append(d)
                if len(samples) >= max(1, n):
                    break
                # 换一帧再检，避免拿到同一个 YOLO 缓存
                time.sleep(0.055)
            else:
                time.sleep(0.03)
        if not samples:
            return None
        centers = np.array([s.center for s in samples], dtype=float)
        med = np.median(centers, axis=0)
        if len(samples) >= 2:
            spread = float(np.max(np.linalg.norm(centers - med, axis=1)))
            stable_px = float(self.cfg["servo"].get("stable_px", 18.0))
            if spread > stable_px:
                self.log.debug("[%s/%s] 检测抖动 %.0fpx，判为不可靠", arm, zone, spread)
                return None
        newest = max(samples, key=lambda s: s.ts)
        area = float(np.median([s.area for s in samples]))
        angle = float(np.median([s.angle for s in samples]))
        return Detection(float(med[0]), float(med[1]), area, angle,
                         newest.bbox, newest.conf, newest.source, newest.ts)

    def scan_for_cube(self, arm: str, zone: str, hover_key: str):
        base = self.planned_q(arm, hover_key)
        for dp in (0.0, -6.0, 6.0, -12.0, 12.0):
            self._check_stop()
            q = base.copy()
            q[PAN] = float(np.clip(q[PAN] + dp, -99.0, 99.0))
            if abs(dp) > 1e-6:
                self.move_q(arm, q, gripper=self.open_g[arm],
                            label=f"{arm} 扫描 pan{dp:+.0f}", speed=0.8,
                            precision=True, tolerate=True, timeout=3.0)
            det = self.detect_stable(arm, zone, force=True, n=2)
            if zone == "box" and not self._box_det_ok(arm, det):
                det = None
            if det is not None:
                self.log.info("[%s] 扫描 pan%+.0f 找到方块: %s", arm, dp, det)
                return q, det
        return None, None

    # ---- 夹爪 ----
    def _measure_air_closed(self):
        grasp = self.cfg["grasp"]
        for a in ("left", "right"):
            arm = self.arms[a]
            try:
                arm.start([self.P[a]["measure"]], gripper=self.open_g[a],
                          label=f"{a} 去 measure", speed_scale=0.6)
                arm.wait(timeout=4.0)
                pos = arm.set_gripper(self.close_g[a], settle=float(
                    self.cfg["motion"]["gripper_close_settle_s"]) + 0.15)
                load = arm.read_load()
                st = self.state[a]
                st.air_closed = float(pos)
                st.load_available = 0.5 < load < 300.0
                st.load_thresh = load + float(grasp["load_min"])
                arm.set_gripper(self.open_g[a], settle=float(
                    self.cfg["motion"]["gripper_open_settle_s"]))
                self._grip[a] = self.open_g[a]
                self.log.info("[%s] 空夹 pos=%.2f load=%.1f 负载可用=%s 阈值=%.1f",
                              a, pos, load, st.load_available, st.load_thresh)
            except C.StopRequested:
                raise
            except Exception as exc:
                self.log.error("[%s] 测空夹失败: %s", a, exc)
                st = self.state[a]
                st.air_closed = float(self.close_g[a])
                st.load_available = False
                st.load_thresh = float(grasp["load_min"])

    def close_and_verify(self, arm: str) -> bool:
        a = self.arms[arm]
        st = self.state[arm]
        grasp = self.cfg["grasp"]
        target = self.close_g[arm]
        cur = float(a.read_gripper())
        step = float(grasp.get("soft_close_step", 8.0))
        while cur > target + 0.5:
            cur = max(target, cur - step)
            a.set_gripper(cur, settle=float(grasp.get("soft_close_settle_s", 0.10)))
        pos = a.set_gripper(target, settle=float(
            self.cfg["motion"]["gripper_close_settle_s"]))
        self._grip[arm] = target
        time.sleep(float(grasp.get("grip_hold_s", 0.25)))
        load = a.read_load()
        time.sleep(float(grasp.get("load_settle_s", 0.20)))
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
        self.arms[arm].set_gripper(self.open_g[arm], settle=float(
            self.cfg["motion"]["gripper_open_settle_s"]))
        self._grip[arm] = self.open_g[arm]
        self.state[arm].holding = False

    def release_here(self, arm: str, back_key: str, *, label: str,
                     speed: float = 0.9, timeout: float = 4.0):
        a = self.arms[arm]
        dwell = float(self.cfg["motion"].get("release_dwell_s", 0.20))
        a.set_gripper(self.open_g[arm], settle=dwell)
        self._grip[arm] = self.open_g[arm]
        self.state[arm].holding = False
        self.move_pose(arm, [back_key], gripper=self.open_g[arm],
                       label=label, speed=speed, timeout=timeout, tolerate=True)

    # ------------------------------------------------------------------
    # 阶段：抓取（桌面 / 盒内共用，全程视觉伺服）
    # ------------------------------------------------------------------
    def pick(self, arm: str, zone: str) -> bool:
        hover_key, pick_key = ZONES[zone]
        approach_key = "box_approach" if zone == "box" else "table_approach"
        zone_lock = "box" if zone == "box" else "table"
        servo = self.servo[arm]
        if zone == "box":
            tmo = float(self.cfg["safety"].get(
                "box_pick_timeout_s", self.cfg["safety"]["phase_timeout_s"]))
        else:
            tmo = float(self.cfg["safety"]["phase_timeout_s"])
        timer = PhaseTimer(tmo, f"{arm} {zone} 抓取")
        retries = max(1, int(self.cfg["servo"].get("pick_retries", 3)))
        self.acquire(arm, zone_lock)
        try:
            self.log.info("[%s] 阶段：%s 抓取", arm, "盒内" if zone == "box" else "桌面")
            self.move_pose(arm, ["carry", approach_key], gripper=self.open_g[arm],
                           label=f"{arm} 去{('盒子' if zone == 'box' else '桌面')}上方",
                           speed=1.0, timeout=6.0, tolerate=True)
            self.move_pose(arm, [hover_key], gripper=self.open_g[arm],
                           label=f"{arm} 观察位", speed=0.9, timeout=4.0, tolerate=True)
            delta = self.vertical_delta[(arm, zone)]
            for attempt in range(1, retries + 1):
                timer.check()
                self._check_stop()
                det_hover = self.detect_stable(arm, zone, force=True, n=2)
                if zone == "box" and not self._box_det_ok(arm, det_hover):
                    det_hover = None
                if det_hover is None:
                    self.log.warning("[%s] %s 观察位没看到方块，扫描", arm, zone)
                    q_scan, det_hover = self.scan_for_cube(arm, zone, hover_key)
                    if det_hover is None:
                        self.log.warning("[%s] %s 仍然找不到方块（第 %d 次）",
                                         arm, zone, attempt)
                        self.move_pose(arm, [hover_key], gripper=self.open_g[arm],
                                       label=f"{arm} 回观察位", speed=0.9,
                                       timeout=4.0, tolerate=True)
                        continue
                    base = q_scan
                else:
                    base = self.planned_q(arm, hover_key)

                hover_px, pick_px = servo.targets(zone)
                J = servo.ensure_jacobian(self, zone, base)
                self.stats["servo_iters"] += 1

                if hover_px is not None and pick_px is not None and J is not None:
                    # 已经学习过：闭环对准到学习到的 hover 像素
                    q_h, det_hover, converged, err = servo.center(
                        self, zone, base, hover_px, J)
                    if det_hover is None:
                        self.log.warning("[%s] %s hover 对准丢目标", arm, zone)
                        self.move_pose(arm, [hover_key], gripper=self.open_g[arm],
                                       label=f"{arm} 回观察位", speed=0.9,
                                       timeout=4.0, tolerate=True)
                        continue
                    if not converged:
                        self.hover_fails[(arm, zone)] += 1
                        self.log.warning("[%s] %s hover 残差 %.0fpx（容差 %.0f）",
                                         arm, zone, err, servo.tol_px)
                        if self.hover_fails[(arm, zone)] >= 2:
                            servo.reset(zone, keep_jacobian=True)
                            self.hover_fails[(arm, zone)] = 0
                            self.log.warning("[%s/%s] 学习值可能过期，已清除重新引导",
                                             arm, zone)
                        limit = float(self.cfg["servo"].get("hover_abort_px", 40.0))
                        if err > limit:
                            # 方块已经超出这条臂的可达范围（常见于方块落在
                            # 盒子另一侧、pan 顶到限位）。不要继续下探把方块
                            # 推走，直接让另一条臂试。
                            raise C.GraspFailed(
                                f"{arm} {zone} hover 偏差 {err:.0f}px，超出可达范围")
                    else:
                        self.hover_fails[(arm, zone)] = 0
                else:
                    # 没有学习值（或雅可比测不出来）：按标定位姿下探。
                    # 只有连目标像素都没有时才做搜索偏移；已经学过目标时
                    # 不应再偏移，否则会把学到的几何关系带偏。
                    q_h = base.copy()
                    if hover_px is None or pick_px is None:
                        dpan, delb = self._bootstrap_offset(attempt)
                        q_h[PAN] = float(np.clip(q_h[PAN] + dpan, -99.0, 99.0))
                        q_h[ELBOW] = float(np.clip(q_h[ELBOW] + delb, -95.0, 95.0))
                        if abs(dpan) > 1e-6 or abs(delb) > 1e-6:
                            self.move_q(arm, q_h, gripper=self.open_g[arm],
                                        label=f"{arm} 引导偏移 {dpan:+.0f}/{delb:+.0f}",
                                        speed=0.8, precision=True, tolerate=True,
                                        timeout=2.5)
                    # 学习值必须是“这一次成功位姿下的像素”，所以偏移后再检一次
                    det_hover = self.detect_stable(arm, zone, force=True, n=2)
                    if det_hover is None:
                        self.log.warning("[%s] %s 引导位仍看不到方块（第 %d 次）",
                                         arm, zone, attempt)
                        self.move_pose(arm, [hover_key], gripper=self.open_g[arm],
                                       label=f"{arm} 回观察位", speed=0.9,
                                       timeout=4.0, tolerate=True)
                        continue

                q_final, det_bottom, _, n_det = servo.descend(
                    self, zone, q_h, delta, hover_px, pick_px, J)
                ok_gate, why = servo.gate_ok(det_bottom, pick_px)
                if not ok_gate:
                    self.stats["gate_fails"] += 1
                    self.gate_fails[(arm, zone)] += 1
                    self.log.warning("[%s] %s 合爪闸门拒绝（第 %d 次）: %s",
                                     arm, zone, attempt, why)
                    if self.gate_fails[(arm, zone)] >= int(
                            self.cfg["servo"].get("gate_reset_after", 3)):
                        servo.reset(zone, keep_jacobian=True)
                        self.gate_fails[(arm, zone)] = 0
                        self.log.warning("[%s/%s] 学习值可能过期，已清除重新引导",
                                         arm, zone)
                    self.move_pose(arm, [hover_key], gripper=self.open_g[arm],
                                   label=f"{arm} 退出盒子", speed=0.9,
                                   timeout=4.0, tolerate=True)
                    continue
                self.log.info("[%s] %s 下探就位: %s（%s）", arm, zone,
                              np.round(det_bottom.center, 0).tolist(), why)
                if self.close_and_verify(arm):
                    self.move_pose(arm, [hover_key], gripper=self.close_g[arm],
                                   label=f"{arm} 夹起", speed=float(
                                       self.cfg["grasp"].get("lift_speed", 0.5)),
                                   timeout=6.0, tolerate=True)
                    if self.lift_verify(arm):
                        servo.learn_success(zone, det_hover.center, det_bottom.center)
                        self.gate_fails[(arm, zone)] = 0
                        self.move_pose(arm, ["carry"], gripper=self.close_g[arm],
                                       label=f"{arm} 搬运位", speed=1.0,
                                       timeout=5.0, tolerate=True)
                        self.state[arm].holding = True
                        return True
                    self.log.warning("[%s] %s 抬起滑脱", arm, zone)
                else:
                    self.log.warning("[%s] %s 第 %d 次没夹住", arm, zone, attempt)
                self.release_gripper(arm)
                self.stats["pick_retries"] += 1
                self.move_pose(arm, [hover_key], gripper=self.open_g[arm],
                               label=f"{arm} 重试回观察位", speed=0.9,
                               timeout=4.0, tolerate=True)
            raise C.GraspFailed(f"{arm} {zone} 抓取 {retries} 次都失败")
        finally:
            self.release(arm, zone_lock)

    @staticmethod
    def _bootstrap_offset(attempt: int) -> tuple[float, float]:
        """首次引导时的小范围搜索偏移（pan, elbow）。

        盒子/桌面方块可能被上一轮放偏，标定位姿不一定正对。按先小后大、
        先 pan 后 elbow 的顺序覆盖常用偏差；合爪前仍然要过视觉闸门。
        """
        table = ((0.0, 0.0), (4.0, 0.0), (-4.0, 0.0), (0.0, 4.0),
                 (0.0, -4.0), (7.0, 4.0), (-7.0, 4.0), (7.0, -4.0),
                 (-7.0, -4.0))
        return table[min(max(0, attempt - 1), len(table) - 1)]

    # ------------------------------------------------------------------
    # 阶段：放入盒子 + 落点复检
    # ------------------------------------------------------------------
    def descend_to(self, arm: str, base_q, delta, *, gripper, label: str,
                   speed: float = 0.5):
        target = np.clip(np.asarray(base_q, float) + np.asarray(delta, float),
                         -100.0, 100.0)
        self.move_q(arm, target, gripper=gripper, label=label, speed=speed,
                    precision=False, tolerate=True, timeout=5.0)
        return target

    def _place_anchor(self, arm: str) -> Optional[np.ndarray]:
        r = self.servo_state.rec(arm, "place")
        a = r.get("anchor_px")
        return np.asarray(a, dtype=float) if a else None

    def _remember_place(self, arm: str, det: Detection) -> None:
        r = self.servo_state.rec(arm, "place")
        old = r.get("anchor_px")
        if old is None:
            r["anchor_px"] = [round(float(v), 2) for v in det.center]
        else:
            mix = 0.2 * det.center + 0.8 * np.asarray(old, dtype=float)
            r["anchor_px"] = [round(float(v), 2) for v in mix]
        self.servo_state.save()

    def verify_in_box(self, arm: str, frame_out=None) -> tuple[str, Optional[Detection]]:
        """返回 ('ok'|'off'|'unknown', det)。unknown 不重放（可能是视角问题）。"""
        if not bool(self.cfg["smooth"].get("place_verify", True)):
            return "ok", None
        det = self.detect_stable(arm, "box", force=True, n=2)
        if det is None:
            self.log.warning("[%s] 盒内复检没看到方块（继续，下一步抓取会验证）", arm)
            frame = self.cams[arm].fresh(max_age=1.0)
            if frame is not None:
                save_snapshot(self.cfg, self.det, frame, None,
                              f"{arm}_place_unverified", self.log)
            return "unknown", None
        anchor = self._place_anchor(arm)
        if anchor is None:
            self._remember_place(arm, det)
            self.log.info("[%s] 落点确认（首次作为锚点）: %s", arm,
                          np.round(det.center, 0).tolist())
            return "ok", det
        err = float(np.abs(det.center - anchor).max())
        tol = float(self.cfg["smooth"].get("place_tol_px", 55.0))
        if err <= tol:
            self._remember_place(arm, det)
            self.log.info("[%s] 落点确认: 偏差 %.0fpx（容差 %.0f）", arm, err, tol)
            return "ok", det
        self.log.warning("[%s] 落点偏差 %.0fpx（容差 %.0f），方块 %s，锚点 %s",
                         arm, err, tol, np.round(det.center, 0).tolist(),
                         np.round(anchor, 0).tolist())
        frame = self.cams[arm].fresh(max_age=1.0)
        if frame is not None:
            save_snapshot(self.cfg, self.det, frame, det,
                          f"{arm}_place_off", self.log)
        return "off", det

    def place_in_box(self, arm: str) -> None:
        timer = PhaseTimer(float(self.cfg["safety"]["phase_timeout_s"]),
                           f"{arm} 放入盒子")
        attempts = 1 + max(0, int(self.cfg["smooth"].get("place_retries", 1)))
        last_reason = ""
        for k in range(attempts):
            timer.check()
            self.acquire(arm, "box")
            try:
                self.log.info("[%s] 阶段：放入盒子", arm)
                self.move_pose(arm, ["carry", "box_approach"], gripper=self.close_g[arm],
                               label=f"{arm} 去盒子上方", speed=1.0,
                               timeout=6.0, tolerate=True)
                self.move_pose(arm, ["box_hover"], gripper=self.close_g[arm],
                               label=f"{arm} 盒口", speed=0.9, timeout=4.0,
                               tolerate=True)
                self.descend_to(arm, self.planned_q(arm, "box_hover"),
                                self.drop_delta[arm], gripper=self.close_g[arm],
                                label=f"{arm} 竖直放入",
                                speed=float(self.cfg["smooth"].get("descent_speed", 0.5)))
                # 原地张爪 + 短 dwell：落点比“边抬边松”稳定
                self.release_here(arm, "box_hover", label=f"{arm} 松爪抬离")
                status, det = self.verify_in_box(arm)
            finally:
                self.release(arm, "box")
            if status != "off":
                # 放完立刻离盒，同时把另一条臂预位到盒口
                self.move_pose(arm, ["box_approach", "carry"], gripper=self.open_g[arm],
                               label=f"{arm} 离开盒子", speed=1.0, timeout=6.0,
                               tolerate=True)
                other = "right" if arm == "left" else "left"
                self._preposition(other, "box_approach")
                return
            last_reason = "落点偏了"
            if k + 1 < attempts:
                self.log.warning("[%s] 重新夹起方块再放一次", arm)
                self.pick(arm, "box")
        raise C.PhaseError(f"{arm} 放入盒子失败: {last_reason}")

    # ------------------------------------------------------------------
    # 状态定位 / 恢复
    # ------------------------------------------------------------------
    def _arm_seems_holding(self, arm: str) -> bool:
        """只信“本次运行里 close+lift 验证过”的持有状态。

        启动时不能用夹爪位置/负载反推：空夹合上时读数本来就在 4~5 附近，
        开爪后读数还要几百毫秒才回到全开位置，再加上负载偶发尖峰，实测
        会把空爪误判成夹着方块，导致整轮状态错位。
        """
        return bool(self.state[arm].holding)

    def _box_ref_px(self, arm: str) -> np.ndarray:
        hp, _ = self.servo[arm].targets("box")
        if hp is not None:
            return hp
        return np.asarray(self.cfg["servo"].get("bootstrap_center_px",
                                                [340.0, 330.0]), dtype=float)

    def _box_det_ok(self, arm: str, det: Optional[Detection]) -> bool:
        """盒内检测的空间合理性检查：方块只可能在盒子附近。

        盒壁/塑料反光会制造几十到几百像素的绿色小斑（实测 187px、
        面积 243）。它们能把雅可比测歪、把下探带偏，所以先按“离盒子
        中心不能太远”过滤掉。
        """
        if det is None:
            return False
        if not bool(self.cfg["servo"].get("box_gate_enable", True)):
            return True
        ref = self._box_ref_px(arm)
        limit = float(self.cfg["servo"].get("box_gate_px", 200.0))
        d = float(np.linalg.norm(det.center - ref))
        if d > limit:
            self.log.debug("[%s] 盒内检测偏离 %.0fpx > %.0f，忽略: %s",
                           arm, d, limit, det)
            return False
        return True

    def locate_cube(self) -> str:
        """判断方块在哪：left/right（夹爪里）、box、table、unknown。

        不能只靠 box_roi 判断：这个 ROI 几乎是整幅画面，臂在 home 时
        左相机同时能看到桌面方块和盒子，单看 ROI 会把桌面方块误判成
        盒内。这里按固定顺序实际去两个观察位看一眼：
          1. 盒内（右臂到 box_hover）——相机对着盒内，看到就是盒内；
          2. 桌面（左臂到 table_hover）——只有盒子空着才会走到这一步。
        全程不合爪、不下探，只是看。
        """
        for arm in ("left", "right"):
            if self._arm_seems_holding(arm):
                self.log.info("定位：%s 夹爪里有方块", arm)
                return arm

        # 1) 盒内
        det = None
        try:
            self.acquire("right", "box")
            try:
                self.move_pose("right", ["carry", "box_approach"],
                               gripper=self.open_g["right"], label="right 定位去盒子",
                               speed=1.0, timeout=6.0, tolerate=True)
                self.move_pose("right", ["box_hover"], gripper=self.open_g["right"],
                               label="right 定位盒口", speed=0.9,
                               timeout=4.0, tolerate=True)
                det = self.detect_stable("right", "box", force=True, n=2)
                if not self._box_det_ok("right", det):
                    det = None
            finally:
                self.release("right", "box")
        except C.StopRequested:
            raise
        except Exception as exc:
            self.log.warning("定位盒内失败: %s", exc)
        if det is not None:
            self.log.info("定位：盒内有方块 %s", det)
            return "box"

        # 2) 桌面
        det = None
        try:
            self.acquire("left", "table")
            try:
                self.move_pose("left", ["carry", "table_approach"],
                               gripper=self.open_g["left"], label="left 定位去桌面",
                               speed=1.0, timeout=6.0, tolerate=True)
                self.move_pose("left", ["table_hover"], gripper=self.open_g["left"],
                               label="left 定位桌面观察位", speed=0.9,
                               timeout=4.0, tolerate=True)
                det = self.detect_stable("left", None, force=True, n=2)
            finally:
                self.release("left", "table")
        except C.StopRequested:
            raise
        except Exception as exc:
            self.log.warning("定位桌面失败: %s", exc)
        if det is not None:
            self.log.info("定位：桌面上有方块 %s", det)
            return "table"

        # 3) 兜底：当前视角再看一遍（可能方块被放在别的角落）
        for arm in ("left", "right"):
            frame = self.cams[arm].fresh(max_age=1.0)
            if frame is None:
                continue
            d = self.det.detect(frame, roi=None, cache_key=f"{arm}:locate",
                                force_yolo=True)
            if d is not None:
                self.log.warning("定位：%s 相机在当前视角看到方块 %s（位置未知）",
                                 arm, d)
                # 当前视角看到的方块，按“桌面”处理最安全：左臂会先用
                # 视觉伺服去桌面位找它，找不到会走扫描而不是盲抓。
                return "table"
        return "unknown"

    def recover(self):
        self._recover_count += 1
        self.log.warning("执行恢复流程")
        for a in ("left", "right"):
            try:
                self.release_gripper(a)
            except Exception:
                pass
        with self._zone_lock:
            self.zone_owner["box"] = None
            self.zone_owner["table"] = None
        for a, cam in self.cams.items():
            if cam.age > 2.5:
                self.log.warning("[%s] 相机 %.1fs 无新帧，重连", a, cam.age)
                cam.reconnect()
        self._sleep(float(self.cfg["safety"]["recover_pause_s"]))
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
        self.cube_state = self.locate_cube()
        self.log.warning("恢复后判定方块位置: %s", self.cube_state)
        if self.state["left"].air_closed is None or self._recover_count % 10 == 0:
            self._measure_air_closed()

    # ------------------------------------------------------------------
    # 一轮
    # ------------------------------------------------------------------
    def pick_auto(self, preferred: str, zone: str) -> str:
        """优先用指定臂抓；它够不到就换另一条臂。

        盒子在两条臂之间的位置并不完全对称：方块掉到某一侧时，那条臂的
        pan 可能顶到 ±99，物理上够不着。以前这种情况只能报错停机；现在
        让另一条臂接手，循环不中断（正常可达时仍然严格按指定臂执行）。
        """
        other = "right" if preferred == "left" else "left"
        try:
            self.pick(preferred, zone)
            return preferred
        except C.StopRequested:
            raise
        except Exception as exc:
            self.log.warning("[%s] %s 抓取失败（%s），尝试另一条臂 %s",
                             preferred, zone, exc, other)
        if self.state[other].holding:
            raise C.GraspFailed(f"{other} 正持有方块，无法换臂")
        self.pick(other, zone)
        return other

    def _holder(self) -> str:
        holders = [a for a in ("left", "right") if self.state[a].holding]
        if not holders:
            raise C.PhaseError("两条臂都没有持有方块")
        if len(holders) > 1:
            self.log.warning("两条臂都报告持有方块，取最新抓取的一条")
        return holders[-1]

    def _cycle_core(self):
        """稳态 4 步（持有臂开始时）：放入 -> 指定臂取出 -> 放回 -> 另一臂取出。

        正常可达时就是需求里的顺序：
            左放入 -> 右取出 -> 右放回 -> 左取出
        某条臂够不到时由 pick_auto 自动换臂，但“取出/放回”的往复结构不变。
        """
        holder = self._holder()
        self.place_in_box(holder)                 # 放入盒内
        arm = self.pick_auto("right", "box")      # 优先右臂取出
        self.place_in_box(arm)                    # 取出臂放回
        arm = self.pick_auto("left", "box")       # 优先左臂取出
        self.cube_state = arm
        self.log.info("[%s] 本轮结束持有方块", arm)

    def run_cycle(self, index: int) -> None:
        t0 = time.perf_counter()
        self.log.info("=" * 66)
        self.log.info("第 %d 轮开始（方块状态=%s）", index, self.cube_state)
        self._check_temp()

        state = self.cube_state
        stale_holder = (state in ("left", "right")
                        and not self.state[state].holding)
        if state == "unknown" or stale_holder:
            state = self.locate_cube()
            self.cube_state = state
        if state == "unknown":
            raise C.PhaseError("看不到方块，无法开始本轮")

        # 先把方块弄到某条臂手里，再跑标准 4 步
        if state == "table":
            state = self.pick_auto("left", "table")   # 首轮引导
            self.cube_state = state
        elif state == "box":
            state = self.pick_auto("right", "box")    # 从盒里取出
            self.cube_state = state
        self._cycle_core()
        dt = time.perf_counter() - t0
        self.log.info("第 %d 轮完成，用时 %.1fs（成功 %d / 失败 %d）",
                      index, dt, self.success, self.failures)

    def shutdown(self):
        """正常停止：把夹着的方块先放回盒子，再开爪下电。"""
        safe = not self.stop_event.is_set()
        for a in ("left", "right"):
            if safe and self.state[a].holding:
                try:
                    self.log.info("[%s] 停机时夹着方块，先放回盒子", a)
                    self.place_in_box(a)
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

    def prepare(self):
        self._measure_air_closed()
        self.cube_state = self.locate_cube()
        for a in ("left", "right"):
            self.log.info("[%s] 竖直增量: table=%s box=%s", a,
                          C.fmt_q(self.vertical_delta[(a, "table")]),
                          C.fmt_q(self.vertical_delta[(a, "box")]))
            self.log.info("[%s] 夹爪: open=%.1f closed=%.1f", a,
                          self.open_g[a], self.close_g[a])
            for z in ZONES:
                hp, pp = self.servo[a].targets(z)
                self.log.info("[%s/%s] 学习值: hover_px=%s pick_px=%s", a, z,
                              None if hp is None else np.round(hp, 0).tolist(),
                              None if pp is None else np.round(pp, 0).tolist())
        self.log.info("启动时方块状态: %s", self.cube_state)
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
        self.csv = CycleCSV(C.LOG_DIR / "cycles_vision.csv")
        self.dual: Optional[DualArm] = None
        self.cams: dict[str, CameraStream] = {}
        self.detector: Optional[Detector] = None
        self.runner: Optional[CycleRunner] = None
        self._watchdog: Optional[threading.Thread] = None
        self._last_progress = time.perf_counter()
        self._cam_reconnect: dict[str, float] = {}

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
                    age = cam.age
                    # CameraStream 自己会一直重连；看门狗只在长时间没帧时
                    # 隔 5 秒补一次 reopen，避免每秒打断它的候选轮换。
                    if age > 5.0:
                        now = time.perf_counter()
                        if now - self._cam_reconnect.get(a, 0.0) > 5.0:
                            self._cam_reconnect[a] = now
                            self.log.warning("[%s] 相机 %.1fs 无新帧，请求重开",
                                             a, age)
                            cam.reconnect()
                streak = self.dual.error_streak() if self.dual else 0
                if streak >= int(self.cfg["safety"]["reconnect_after_errors"]):
                    err_count += 1
                    self.log.error("总线错误累积 %d，重连 (%d/2)", streak, err_count)
                    try:
                        self.dual.reconnect()
                    except Exception as exc:
                        self.log.error("重连失败: %s", exc)
                    if self.dual is not None:
                        for a in ("left", "right"):
                            try:
                                self.dual.arms[a].bus._errors = 0
                            except Exception:
                                pass
                    if err_count >= 2:
                        time.sleep(5.0)
                        err_count = 0
                stall = float(self.cfg["safety"].get("stall_restart_s", 180.0))
                if stall > 0 and time.perf_counter() - self._last_progress > stall:
                    self.log.error("主循环 %.0fs 没有进展，退出让 systemd 重启", stall)
                    try:
                        if self.dual is not None:
                            self.dual.disable_torque()
                    except Exception:
                        pass
                    os._exit(3)
            except Exception as exc:
                self.log.debug("看门狗异常: %s", exc)

    def _validate(self):
        for a in ("left", "right"):
            missing = C.missing_poses(self.cfg, a)
            if missing:
                raise SystemExit(f"[{a}] 缺少 {len(missing)} 个位姿: {missing}\n"
                                 f"请先运行: python calibrate2arm.py --arm {a}")
            ac = C.arm_cfg(self.cfg, a)
            if ac.get("gripper_open") is None or ac.get("gripper_closed") is None:
                raise SystemExit(f"[{a}] 夹爪开合未标定: python calibrate2arm.py --arm {a}")
            bad = C.pose_saturation_report(self.cfg, a)
            if bad:
                raise SystemExit(f"[{a}] 这些位姿被 ±100 截断:\n" + "\n".join(bad)
                                 + f"\n请重标: python calibrate2arm.py --arm {a}")
            if (ac.get("gains") or {}).get("box_roi") is None:
                raise SystemExit(f"[{a}] 没有 box_roi，拒绝启动。"
                                 f"请重标 box_hover/box_drop: "
                                 f"python calibrate2arm.py --arm {a}")

    def _setup(self):
        self._validate()
        self.dual = DualArm(self.cfg, self.log, self.stop_event)
        self.dual.connect()
        for a in ("left", "right"):
            bus = getattr(self.dual.arms[a].bus, "bus", None)
            cal = getattr(bus, "calibration", None) if bus is not None else None
            if not cal:
                continue
            differs, unknown = C.pose_calib_mismatch(self.cfg, a, cal)
            if differs:
                raise SystemExit(f"[{a}] 点位与当前电机标定不一致: {differs}\n"
                                 f"请完整重标: python calibrate2arm.py --arm {a}")
            if unknown:
                raise SystemExit(f"[{a}] 点位缺标定指纹: {unknown}\n"
                                 f"请完整重标: python calibrate2arm.py --arm {a}")
            tight = C.tight_pose_report(self.cfg, a, cal)
            if tight:
                self.log.warning("[%s] 点位离标定边界过近（可能被截断）:\n%s",
                                 a, "\n".join(tight))
        for a in ("left", "right"):
            cands = cam_candidates(self.cfg, a)
            cam = CameraStream(cands[0], logger=self.log, name=a, devices=cands[1:],
                               usb_port=self.cfg["devices"].get(f"{a}_camera_usb"),
                               width=self.cfg["devices"]["camera_width"],
                               height=self.cfg["devices"]["camera_height"],
                               fps=self.cfg["devices"]["camera_fps"])
            cam.start()
            self.cams[a] = cam
        self.detector = Detector(self.cfg, self.log)
        self.runner = CycleRunner(self.cfg, self.dual, self.cams, self.detector,
                                  self.stop_event, self.log, self.csv)
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
            start: str = "auto"):
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

        if start != "auto" and self.runner is not None:
            self.runner.cube_state = start
            self.log.info("按参数指定起始状态: %s", start)

        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True,
                                          name="watchdog")
        self._watchdog.start()
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
                state_before = self.runner.cube_state if self.runner else "?"
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
                state_after = self.runner.cube_state if self.runner else "?"
                note = (f"servo_iters={self.runner.stats['servo_iters']} "
                        f"gate_fails={self.runner.stats['gate_fails']} "
                        f"pick_retries={self.runner.stats['pick_retries']}")
                self.csv.write(cycle, ok, dt, phase, err, state_before, state_after, note)
                self.log.info("统计: 成功 %d / 失败 %d / 连续失败 %d，本轮 %.1fs | %s",
                              self.runner.success, self.runner.failures,
                              consecutive, dt, note)
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
            det = self.detector.detect(frame, roi=None, cache_key=f"{a}:snap",
                                       force_yolo=True) if self.detector else None
            save_snapshot(self.cfg, self.detector, frame, det,
                          f"cycle{cycle:05d}_{a}", self.log)
        self._prune_snapshots()

    def _prune_snapshots(self):
        keep = int(self.cfg["loop"].get("snapshot_keep", 200) or 0)
        if keep <= 0:
            return
        try:
            d = C.HERE / str(self.cfg["loop"].get("snapshot_dir", "snapshots"))
            files = sorted(d.glob("cycle*.jpg"))
            for f in files[:-keep]:
                f.unlink(missing_ok=True)
        except Exception as exc:
            self.log.debug("清理快照失败: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_check(cfg: dict, logger) -> int:
    print("=" * 70)
    print("双臂视觉伺服循环 —— 只读体检（不动电机）")
    print("=" * 70)
    problems = 0

    print("\n[1] 位姿 / 夹爪 / box_roi")
    for a in ("left", "right"):
        missing = C.missing_poses(cfg, a)
        if missing:
            print(f"  [{a}] 缺少位姿: {missing}")
            problems += 1
        sat = C.pose_saturation_report(cfg, a)
        if sat:
            print("\n".join(sat))
            problems += 1
        ac = C.arm_cfg(cfg, a)
        if ac.get("gripper_open") is None or ac.get("gripper_closed") is None:
            print(f"  [{a}] 夹爪未标定")
            problems += 1
        roi = (ac.get("gains") or {}).get("box_roi")
        print(f"  [{a}] box_roi={roi}")
        print(f"  [{a}] 竖直增量 table={C.fmt_q(C.get_pose(cfg, a, 'table_pick') - C.get_pose(cfg, a, 'table_hover'))}")
        print(f"  [{a}] 竖直增量 box  ={C.fmt_q(C.get_pose(cfg, a, 'box_pick') - C.get_pose(cfg, a, 'box_hover'))}")

    print("\n[2] 学习值 servo_state.json")
    st = ServoState(C.HERE / str(cfg["servo"].get("state_file", "servo_state.json")))
    for a in ("left", "right"):
        for z in list(ZONES) + ["place"]:
            r = st.rec(a, z)
            print(f"  [{a}/{z}] hover_px={r.get('hover_px')} pick_px={r.get('pick_px')} "
                  f"anchor_px={r.get('anchor_px')} n={r.get('n')} "
                  f"J={r.get('jacobian')}")

    print("\n[3] 串口")
    try:
        counts = probe_port_counts()
        print(f"  端口电机数: {counts}")
        ports = resolve_arm_ports(cfg, logger)
        print(f"  解析结果: {ports}")
    except Exception as exc:
        print(f"  串口探测失败: {exc}")
        problems += 1

    print("\n[4] 相机 + YOLO")
    try:
        det = Detector(cfg, logger)
        print(f"  YOLO: {'已加载' if det.use_yolo else '未加载，仅 HSV'}")
        for a in ("left", "right"):
            cands = cam_candidates(cfg, a)
            cam = CameraStream(cands[0], devices=cands[1:], logger=logger,
                               name=f"check-{a}",
                               usb_port=cfg["devices"].get(f"{a}_camera_usb"))
            cam.start()
            time.sleep(1.0)
            frame = cam.fresh(max_age=1.5)
            if frame is None:
                print(f"  [{a}] 相机无帧: {cands}")
                problems += 1
            else:
                roi = box_roi_from_cfg(cfg, a)
                d_all = det.detect(frame, roi=None, cache_key=f"chk{a}", force_yolo=True)
                d_roi = det.detect(frame, roi=roi, cache_key=f"chkr{a}", force_yolo=True)
                print(f"  [{a}] 相机 OK {frame.shape} 全画面={d_all} 盒内={d_roi}")
                cv2_ok = save_snapshot(cfg, det, frame, d_all, f"check_{a}", logger)
                if cv2_ok:
                    print(f"  [{a}] 快照: {cv2_ok}")
            cam.stop()
        det.close()
    except Exception as exc:
        print(f"  相机/YOLO 检查失败: {exc}")
        problems += 1

    print("\n[5] 参数")
    sm = cfg["smooth"]
    sv = cfg["servo"]
    print(f"  motion: settle={sm['settle_s']} arrive_tol={sm['arrive_tol']} "
          f"arrive_timeout={sm['arrive_timeout_s']} release_dwell={sm['release_dwell_s']}")
    print(f"  servo : steps={sv['descent_steps']} speed={sv['descent_speed']} "
          f"tol={sv['target_tol_px']}px gate={sv['pick_gate_px']}px "
          f"require_visual={sv['require_visual_before_close']}")

    print("\n" + "=" * 70)
    print("体检结果:", "通过" if problems == 0 else f"{problems} 个问题")
    return 0 if problems == 0 else 1


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
        print("两臂已回 home")
    finally:
        try:
            dual.disable_torque()
        except Exception:
            pass
        dual.stop_workers()
        dual.disconnect()
    return 0


def cmd_reset_servo(cfg: dict, arm: Optional[str], logger) -> int:
    st = ServoState(C.HERE / str(cfg["servo"].get("state_file", "servo_state.json")))
    arms = [arm] if arm else ["left", "right"]
    for a in arms:
        for z in list(ZONES) + ["place"]:
            r = st.rec(a, z)
            r["hover_px"] = None
            r["pick_px"] = None
            r["anchor_px"] = None
            r["jacobian"] = None
            r["n"] = 0
    st.save()
    print(f"已清空学习值: {arms} -> {st.path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="双臂 YOLO 视觉伺服循环夹放")
    ap.add_argument("--config", default=str(C.CONFIG_PATH))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="只读体检")
    sub.add_parser("home", help="两臂回 home")
    p_reset = sub.add_parser("reset-servo", help="清空学习值")
    p_reset.add_argument("--arm", choices=["left", "right"], default=None)

    p_run = sub.add_parser("run", help="运行循环")
    p_run.add_argument("--cycles", type=int, default=0, help="0=无限")
    p_run.add_argument("--seconds", type=float, default=0.0, help="0=不限时")
    p_run.add_argument("--start", default="auto",
                       choices=["auto", "table", "box", "left", "right"],
                       help="起始方块状态（auto=自动定位）")

    args = ap.parse_args()
    cfg = C.load_config(Path(args.config))
    apply_smooth(cfg)
    logger = C.setup_logging(cfg, name="s2b2_vision")

    if args.cmd == "check":
        return cmd_check(cfg, logger)
    if args.cmd == "home":
        return cmd_home(cfg, logger)
    if args.cmd == "reset-servo":
        return cmd_reset_servo(cfg, args.arm, logger)

    sup = Supervisor(cfg, logger)
    sup.run(max_cycles=args.cycles, max_seconds=args.seconds, start=args.start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
