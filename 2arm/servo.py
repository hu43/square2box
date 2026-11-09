#!/usr/bin/env python
"""servo.py — 眼在手上的 YOLO 像素伺服（2-DOF: shoulder_pan + elbow_flex）。

为什么是 pan + elbow：
  * 下探由 shoulder_lift / elbow_flex 的组合完成，lift 必须单调向下；
  * 用 pan + elbow 做水平修正，不会把臂重新抬起，也不会额外加深；
  * 相机和夹爪都刚性固定在腕部，同一高度下像素误差就是水平误差，
    闭环一定收敛。

为什么不依赖手标像素：
  * calib/right_table_hover.jpg 只检出一个 181px 的误检（旧 align_px 就是它），
    calib/left_box_hover.jpg 是 NO TARGET。手标像素不可信；
  * hover_px / pick_px 在还没有学习值时用标定位姿引导，抓成功一次后
    自动学习并持久化到 servo_state.json，后续全程闭环。

安全约束：
  * 修正只写 pan / elbow，且单步 ≤ max_step_deg、累计 ≤ max_total_deg；
  * 下探按标定增量分段，绝不因为视觉而多下探；
  * 合爪前必须有一次新鲜、稳定的检测落在 pick_px 容差内，否则不合爪。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

import armcore as C

if TYPE_CHECKING:  # 只用于类型提示，避免和 armvision 互相 import
    from armvision import Detection


ZONES = {
    "table": ("table_hover", "table_pick"),
    "box": ("box_hover", "box_pick"),
}

PAN = C.JOINTS.index("shoulder_pan")
ELBOW = C.JOINTS.index("elbow_flex")


class ServoState:
    """学习值持久化：servo_state.json（和 config.json 分开，避免改标定）。"""

    VERSION = 1

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = {"version": self.VERSION, "arms": {}}
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                if isinstance(raw, dict) and isinstance(raw.get("arms"), dict):
                    self.data = raw
                    self.data.setdefault("version", self.VERSION)
                    self.data.setdefault("arms", {})
            except Exception:
                self.data = {"version": self.VERSION, "arms": {}}

    def rec(self, arm: str, zone: str) -> dict:
        arms = self.data.setdefault("arms", {})
        a = arms.setdefault(arm, {})
        r = a.setdefault(zone, {})
        r.setdefault("hover_px", None)
        r.setdefault("pick_px", None)
        r.setdefault("jacobian", None)
        r.setdefault("n", 0)
        r.setdefault("updated", 0.0)
        return r

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))
            os.replace(tmp, self.path)
        except Exception:
            pass


class PixelServo:
    """单臂的像素伺服。runner 需要提供下面几个方法：

        runner.move_q(arm, q, *, label="", speed=1.0, precision=False,
                      tolerate=False, timeout=None) -> bool
        runner.detect(arm, zone, *, force=False) -> Detection | None
        runner.detect_stable(arm, zone, *, force=False, n=2,
                             max_age=...) -> Detection | None
        runner.pose(arm, key) -> np.ndarray
        runner.log  (logging.Logger)
    """

    def __init__(self, cfg: dict, arm: str, logger, state: Optional[ServoState] = None):
        self.cfg = cfg
        self.arm = arm
        self.log = logger
        sc = cfg["servo"]
        self.enabled = bool(sc.get("enable", True))
        self.probe_deg = float(sc.get("probe_deg", 2.5))
        self.tol_px = float(sc.get("target_tol_px", 7.0))
        self.gate_px = float(sc.get("pick_gate_px", 30.0))
        self.gate_fresh_s = float(sc.get("pick_gate_fresh_s", 0.35))
        self.max_iters = int(sc.get("max_iters", 6))
        self.max_step_deg = float(sc.get("max_step_deg", 5.0))
        self.max_total_deg = float(sc.get("max_total_deg", 12.0))
        self.damping = float(sc.get("damping", 0.15))
        self.steps = int(sc.get("descent_steps", 4))
        self.descent_tol_px = float(sc.get("descent_tol_px", 12.0))
        self.descent_speed = float(sc.get("descent_speed", 0.45))
        self.learn_ema = float(sc.get("learn_ema", 0.25))
        self.learn_max_jump_px = float(sc.get("learn_max_jump_px", 45.0))
        self.reject_jump_px = float(sc.get("reject_jump_px", 90.0))
        self.reject_area_ratio = float(sc.get("reject_area_ratio", 2.5))
        self.require_visual = bool(sc.get("require_visual_before_close", True))
        self.bootstrap_center = np.asarray(
            sc.get("bootstrap_center_px", [320.0, 300.0]), dtype=float)
        self.bootstrap_radius = float(sc.get("bootstrap_radius_px", 170.0))
        self.state = state or ServoState(C.HERE / str(sc.get("state_file", "servo_state.json")))

    # ------------------------------------------------------------------
    # 学习值
    # ------------------------------------------------------------------
    def targets(self, zone: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        r = self.state.rec(self.arm, zone)
        hp = r.get("hover_px")
        pp = r.get("pick_px")
        return (np.asarray(hp, dtype=float) if hp else None,
                np.asarray(pp, dtype=float) if pp else None)

    def has_targets(self, zone: str) -> bool:
        hp, pp = self.targets(zone)
        return hp is not None and pp is not None

    def jacobian(self, zone: str) -> Optional[np.ndarray]:
        j = self.state.rec(self.arm, zone).get("jacobian")
        if not j:
            return None
        try:
            arr = np.asarray(j, dtype=float)
        except Exception:
            return None
        if arr.shape != (2, 2) or not np.all(np.isfinite(arr)):
            return None
        # 缓存值也要过一遍合理性检查：上一版可能把反光测出的坏雅可比存了
        # 进去（pan 增益只有 0.36 px/deg），不校验会一直用下去。
        min_pan = float(self.cfg["servo"].get("jacobian_min_pan_gain", 1.0))
        min_det = float(self.cfg["servo"].get("jacobian_min_det", 0.5))
        if abs(float(np.linalg.det(arr))) < min_det or abs(float(arr[0, 0])) < min_pan:
            self.log.warning("[%s/%s] 缓存雅可比不可信，丢弃: %s",
                             self.arm, zone, np.round(arr, 2).tolist())
            r = self.state.rec(self.arm, zone)
            r["jacobian"] = None
            self.state.save()
            return None
        return arr

    def reset(self, zone: str, keep_jacobian: bool = True) -> None:
        r = self.state.rec(self.arm, zone)
        self.log.warning("[%s/%s] 清除学习值（重新引导）", self.arm, zone)
        r["hover_px"] = None
        r["pick_px"] = None
        r["n"] = 0
        if not keep_jacobian:
            r["jacobian"] = None
        self.state.save()

    def learn_success(self, zone: str, hover_px: Optional[np.ndarray],
                      pick_px: Optional[np.ndarray]) -> None:
        r = self.state.rec(self.arm, zone)
        r["n"] = int(r.get("n") or 0) + 1
        r["updated"] = time.time()
        if pick_px is not None:
            old = r.get("pick_px")
            new = np.asarray(pick_px, dtype=float)
            if old is None:
                r["pick_px"] = [round(float(v), 2) for v in new]
            else:
                old = np.asarray(old, dtype=float)
                if float(np.linalg.norm(new - old)) <= self.learn_max_jump_px:
                    mix = self.learn_ema * new + (1.0 - self.learn_ema) * old
                    r["pick_px"] = [round(float(v), 2) for v in mix]
        # hover_px 只在第一次成功时学习：伺服后的 hover_px 已经是目标，
        # 不能拿“修正前”的检测覆盖它。
        if r.get("hover_px") is None and hover_px is not None:
            r["hover_px"] = [round(float(v), 2) for v in np.asarray(hover_px)]
        self.state.save()
        self.log.info("[%s/%s] 学习: hover_px=%s pick_px=%s n=%s",
                      self.arm, zone,
                      None if r.get("hover_px") is None else np.round(r["hover_px"], 0).tolist(),
                      None if r.get("pick_px") is None else np.round(r["pick_px"], 0).tolist(),
                      r["n"])

    # ------------------------------------------------------------------
    # 雅可比
    # ------------------------------------------------------------------
    def ensure_jacobian(self, runner, zone: str, base_q: np.ndarray) -> Optional[np.ndarray]:
        """在 hover 位对 pan / elbow 各做一次微探针，测 d(px,py)/d(deg)。"""
        J = self.jacobian(zone)
        if J is not None:
            return J
        if not self.enabled:
            return None
        base_q = np.asarray(base_q, dtype=float).copy()
        det = runner.detect_stable(self.arm, zone, force=True, n=2)
        if det is None:
            self.log.warning("[%s/%s] 无法测雅可比：hover 位看不到方块", self.arm, zone)
            return None
        p0 = det.center
        J = np.zeros((2, 2), dtype=float)
        ok = True
        try:
            for col, idx in ((0, PAN), (1, ELBOW)):
                q = base_q.copy()
                q[idx] = float(np.clip(q[idx] + self.probe_deg, -99.0, 99.0))
                if abs(q[idx] - base_q[idx]) < 1e-6:
                    ok = False
                    break
                runner.move_q(self.arm, q, label=f"{self.arm}/{zone} 测雅可比 {col}",
                              speed=0.6, precision=True, tolerate=True, timeout=2.0)
                d2 = runner.detect_stable(self.arm, zone, force=True, n=2)
                if d2 is None:
                    ok = False
                    break
                J[:, col] = (d2.center - p0) / float(q[idx] - base_q[idx])
                runner.move_q(self.arm, base_q,
                              label=f"{self.arm}/{zone} 雅可比复位",
                              speed=0.7, precision=True, tolerate=True, timeout=2.0)
        finally:
            # 无论测量成功与否都回到 hover，避免把臂留在探针位。
            if not np.allclose(base_q, runner.planned_q(self.arm), atol=0.05):
                runner.move_q(self.arm, base_q, label=f"{self.arm}/{zone} 回 hover",
                              speed=0.8, precision=True, tolerate=True, timeout=2.0)
        if not ok:
            self.log.warning("[%s/%s] 雅可比测量失败（探针中丢检测），本次不做闭环",
                             self.arm, zone)
            return None
        # 合理性检查：pan 必须真的能推动像素，行列式不能接近 0。
        # 盒壁反光造成的小绿斑也会“测出”雅可比，但 pan 增益只有 0.3~0.4
        # px/deg（真方块通常是 2~3），靠这个把它挡掉。
        min_pan = float(self.cfg["servo"].get("jacobian_min_pan_gain", 1.0))
        min_det = float(self.cfg["servo"].get("jacobian_min_det", 0.5))
        if (not np.all(np.isfinite(J)) or float(np.abs(J).max()) > 80.0
                or abs(float(np.linalg.det(J))) < min_det
                or abs(float(J[0, 0])) < min_pan):
            self.log.warning("[%s/%s] 雅可比不可信: %s", self.arm, zone,
                             np.round(J, 2).tolist())
            return None
        r = self.state.rec(self.arm, zone)
        r["jacobian"] = [[round(float(v), 4) for v in row] for row in J]
        r["updated"] = time.time()
        self.state.save()
        self.log.info("[%s/%s] 雅可比已测: pan=(%.2f,%.2f) elbow=(%.2f,%.2f) px/deg",
                      self.arm, zone, J[0, 0], J[1, 0], J[0, 1], J[1, 1])
        return J

    def _solve(self, J: np.ndarray, err: np.ndarray) -> np.ndarray:
        # 阻尼最小二乘：dq = -J^T (J J^T + λ²I)^-1 e
        A = J @ J.T + (self.damping ** 2) * np.eye(2)
        try:
            dq = -J.T @ np.linalg.solve(A, err)
        except np.linalg.LinAlgError:
            dq = -J.T @ np.linalg.pinv(A) @ err
        dq = np.asarray(dq, dtype=float)
        if not np.all(np.isfinite(dq)):
            return np.zeros(2)
        m = float(np.abs(dq).max())
        if m > self.max_step_deg:
            dq *= self.max_step_deg / m
        return dq

    # ------------------------------------------------------------------
    # hover 对准
    # ------------------------------------------------------------------
    def center(self, runner, zone: str, q0: np.ndarray,
               target_px: np.ndarray,
               J: Optional[np.ndarray]) -> tuple[np.ndarray, Optional["Detection"], bool, float]:
        """把方块像素逼到 target_px。返回 (q, 最后检测, 是否收敛, 残差px)。"""
        q = np.asarray(q0, dtype=float).copy()
        target = np.asarray(target_px, dtype=float)
        if J is None:
            det = runner.detect_stable(self.arm, zone, force=True, n=2)
            err = -1.0 if det is None else float(np.abs(det.center - target).max())
            return q, det, det is not None, err
        total = np.zeros(2, dtype=float)
        det = None
        lost = 0
        for it in range(self.max_iters + 1):
            det = runner.detect_stable(self.arm, zone, force=True, n=2)
            if det is None:
                lost += 1
                if lost >= 2:
                    return q, None, False, -1.0
                continue
            lost = 0
            err = det.center - target
            emax = float(np.abs(err).max())
            if emax <= self.tol_px:
                self.log.info("[%s/%s] hover 对准收敛: 残差 %.0fpx (第 %d 步)",
                              self.arm, zone, emax, it)
                return q, det, True, emax
            if it >= self.max_iters:
                break
            dq = self._solve(J, err)
            # 累计修正限幅，避免伺服把臂带离标定位姿太远
            new_total = np.clip(total + dq, -self.max_total_deg, self.max_total_deg)
            dq = new_total - total
            total = new_total
            q2 = q.copy()
            q2[PAN] = float(np.clip(q2[PAN] + dq[0], -99.0, 99.0))
            q2[ELBOW] = float(np.clip(q2[ELBOW] + dq[1], -95.0, 95.0))
            if float(np.abs(q2 - q).max()) < 0.05:
                break

            runner.move_q(self.arm, q2, label=f"{self.arm}/{zone} hover 对准 {it + 1}",
                          speed=0.8, precision=True, tolerate=True, timeout=2.5)
            q = q2
        det = runner.detect_stable(self.arm, zone, force=True, n=2)
        err = -1.0 if det is None else float(np.abs(det.center - target).max())
        self.log.warning("[%s/%s] hover 对准未完全收敛: 残差 %.0fpx（容差 %.0f）",
                         self.arm, zone, err, self.tol_px)
        return q, det, False, err

    # ------------------------------------------------------------------
    # 分段下探 + 跟踪
    # ------------------------------------------------------------------
    def descend(self, runner, zone: str, q_hover: np.ndarray, delta: np.ndarray,
                hover_px: Optional[np.ndarray], pick_px: Optional[np.ndarray],
                J: Optional[np.ndarray], steps: Optional[int] = None,
                ) -> tuple[np.ndarray, Optional["Detection"], np.ndarray, int]:
        """沿标定增量分段下探。

        返回 (q_final, 最后检测, 规划终点, 检测次数)。
        每段后重新检测：目标像素按段号从 hover_px 线性过渡到 pick_px，
        残差用 pan/elbow 修正；lift 只按规划值走，绝不因视觉额外加深。
        """
        q0 = np.asarray(q_hover, dtype=float).copy()
        delta = np.asarray(delta, dtype=float)
        n = int(steps or self.steps)
        n = max(1, n)
        track = (hover_px is not None and pick_px is not None and J is not None)
        last: Optional["Detection"] = None
        prev: Optional["Detection"] = None
        n_det = 0
        corr = np.zeros(C.NJ, dtype=float)   # pan/elbow 的累计修正
        for k in range(1, n + 1):
            q_plan = np.clip(q0 + delta * (k / n) + corr, -100.0, 100.0)
            runner.move_q(self.arm, q_plan,
                          label=f"{self.arm}/{zone} 下探 {k}/{n}",
                          speed=self.descent_speed, precision=False,
                          tolerate=True, timeout=3.0)
            # 最后一段要“新检”，中间段允许用最近缓存以省时间
            force = (k == n)
            n_sample = 2 if force else 1
            det = runner.detect_stable(self.arm, zone, force=force, n=n_sample)
            n_det += 1
            target = None
            if track:
                target = (np.asarray(hover_px)
                          + (np.asarray(pick_px) - np.asarray(hover_px)) * (k / n))
            if det is not None and not self._plausible(det, prev, target):
                self.log.warning("[%s/%s] 下探第 %d 段检测异常，忽略: %s",
                                 self.arm, zone, k, det)
                det = None
            if det is not None:
                prev = det
                last = det
            if not track or det is None:
                continue
            err = det.center - target
            if float(np.abs(err).max()) <= self.descent_tol_px:
                continue
            dq = self._solve(J, err)
            # 修正量累计后限幅：既不会把臂带离规划路径太远，也不会因为
            # 下一步重新从 q0 计算而丢掉上一步的修正。
            new_pan = float(np.clip(corr[PAN] + dq[0],
                                    -self.max_total_deg, self.max_total_deg))
            new_elb = float(np.clip(corr[ELBOW] + dq[1],
                                    -self.max_total_deg, self.max_total_deg))
            dq2 = np.array([new_pan - corr[PAN], new_elb - corr[ELBOW]])
            if float(np.abs(dq2).max()) < 0.05:
                continue
            corr[PAN] = new_pan
            corr[ELBOW] = new_elb
            q2 = np.clip(q0 + delta * (k / n) + corr, -100.0, 100.0)
            runner.move_q(self.arm, q2,
                          label=f"{self.arm}/{zone} 下探跟踪 {k}/{n}",
                          speed=self.descent_speed, precision=False,
                          tolerate=True, timeout=2.0)
            d2 = runner.detect_stable(self.arm, zone, force=True, n=1)
            n_det += 1
            if d2 is not None and self._plausible(d2, prev, target):
                prev = d2
                last = d2
        return (np.clip(q0 + delta + corr, -100.0, 100.0), last,
                np.clip(q0 + delta, -100.0, 100.0), n_det)

    def _plausible(self, det: "Detection", prev: Optional["Detection"],
                   expected_px: Optional[np.ndarray]) -> bool:
        """拒绝反光/盒壁造成的跳变检测。"""
        if not np.all(np.isfinite(det.center)):
            return False
        if expected_px is not None:
            if float(np.abs(det.center - expected_px).max()) > self.reject_jump_px:
                return False
        if prev is not None:
            if (prev.area > 1.0 and det.area > 1.0
                    and (det.area / prev.area > self.reject_area_ratio
                         or prev.area / det.area > self.reject_area_ratio)
                    and float(np.linalg.norm(det.center - prev.center)) > 40.0):
                return False
        return True

    # ------------------------------------------------------------------
    # 合爪闸门
    # ------------------------------------------------------------------
    def gate_ok(self, det: Optional["Detection"], pick_px: Optional[np.ndarray],
                now: Optional[float] = None) -> tuple[bool, str]:
        """合爪前必须满足：有检测、足够新、位于抓取像素附近。"""
        if not self.require_visual:
            return True, ""
        if det is None:
            return False, "下探到底没有检测到方块（拒绝盲合）"
        now = time.time() if now is None else now
        age = max(0.0, now - float(getattr(det, "ts", now)))
        if age > self.gate_fresh_s:
            return False, f"底部检测已过期 {age:.2f}s（拒绝盲合）"
        if pick_px is not None:
            err = float(np.abs(det.center - np.asarray(pick_px)).max())
            if err > self.gate_px:
                return False, f"底部偏差 {err:.0f}px > {self.gate_px:.0f}px（拒绝盲合）"
            return True, f"底部偏差 {err:.0f}px"
        # 首次引导：还没有 pick_px，用画面中心附近作为合理性判断
        err = float(np.linalg.norm(det.center - self.bootstrap_center))
        if err > self.bootstrap_radius:
            return False, (f"首次引导检测 {np.round(det.center, 0).tolist()} "
                           f"偏离引导中心 {err:.0f}px > {self.bootstrap_radius:.0f}px，"
                           f"拒绝合爪")
        return True, f"首次引导 {err:.0f}px @{np.round(det.center, 0).tolist()}"
