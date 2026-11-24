#!/usr/bin/env python
"""handoff_core — 双臂手递循环的公共层（无摄像头）。

两个“交接点”：
  left_front  = 左臂自己的 table_pick（左臂正前方）
  right_front = 右臂自己的 table_pick（右臂正前方）

每条臂都要能在两个交接点上取放，因此每条臂有两组关节位姿：

    left  : left_front  -> table_pick
            right_front -> place_right   （cross 位，由 calibrate_handoff.py 示教）
    right : right_front -> table_pick
            left_front  -> place_left    （cross 位，由 calibrate_handoff.py 示教）

cross 位必须由人把“另一条臂的夹爪”摆到对方 table_pick 的位置、按对方
抓取高度记录。不同臂的关节坐标不能互换，不能用对方 arm 的 table_pick
直接发给自己。

本模块只依赖 armcore / armdriver，不导入 armvision、cv2 或 ultralytics。
"""

from __future__ import annotations

import signal
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np

import armcore as C
from armdriver import (
    DualArm,
    FeetechMotorsBus,
    calibration_from_bus,
    make_motors,
    resolve_arm_ports,
)

FRONTS = ("left", "right")
OWN_KEY = "table_pick"
CROSS_KEY = {"left": "place_right", "right": "place_left"}
CROSS_FRONT = {"left": "right", "right": "left"}


def opposite(front: str) -> str:
    return "right" if front == "left" else "left"


def spot_key(arm: str, front: str) -> str:
    """该臂在某个交接点使用的位姿名。"""
    return OWN_KEY if front == arm else CROSS_KEY[arm]


def spot_pose(cfg: dict, arm: str, front: str) -> np.ndarray:
    return C.get_pose(cfg, arm, spot_key(arm, front))


def own_pose(cfg: dict, arm: str) -> np.ndarray:
    return C.get_pose(cfg, arm, OWN_KEY)


def cross_pose(cfg: dict, arm: str) -> np.ndarray:
    return C.get_pose(cfg, arm, CROSS_KEY[arm])


def pose_margins(q) -> list[tuple[str, float]]:
    q = np.asarray(q, dtype=float)
    return [(C.JOINTS[i], min(float(q[i]) + 100.0, 100.0 - float(q[i])))
            for i in range(C.NJ)]


def validate_config(cfg: dict, require_cross: bool = True,
                    arms=("left", "right")) -> tuple[list[str], list[str]]:
    """配置层校验。返回 (阻断问题, 警告)。"""
    problems: list[str] = []
    warnings: list[str] = []

    for arm in arms:
        missing = C.missing_poses(cfg, arm)
        if missing:
            problems.append(
                f"[{arm}] 缺少 {len(missing)} 个位姿: {', '.join(missing)}\n"
                f"  请先运行: python ../2arm/calibrate2arm.py --arm {arm}")
        for which in ("open", "closed"):
            if cfg["arms"][arm].get(f"gripper_{which}") is None:
                problems.append(
                    f"[{arm}] 缺少 gripper_{which}，请先运行: "
                    f"python ../2arm/calibrate2arm.py --arm {arm}")

        problems.extend(C.pose_saturation_report(cfg, arm))
        warnings.extend(C.duplicate_pose_report(cfg, arm))
        warnings.extend(C.pose_delta_report(cfg, arm))

        if not require_cross:
            continue

        key = CROSS_KEY[arm]
        q = (cfg["arms"][arm].get("poses") or {}).get(key)
        if q is None:
            problems.append(
                f"[{arm}] 缺少交叉位 '{key}'（{arm} 臂到 {CROSS_FRONT[arm]} 前方）。\n"
                f"  请先运行一次性示教: python calibrate_handoff.py --arm {arm}")
            continue

        q = np.asarray(q, dtype=float)
        bad = C.saturated_joints(q)
        if bad:
            problems.append(
                f"[{arm}] 交叉位 '{key}' 顶到 ±100（{', '.join(bad)}），"
                f"该点不可达/不可用；请把交接点往该臂方向挪一点后重录。")
        margins = pose_margins(q)
        tight = [f"{n}({m:.1f})" for n, m in margins if m < 2.0]
        if tight:
            warnings.append(
                f"[{arm}] 交叉位 '{key}' 离标定边界很近: {', '.join(tight)}；"
                f"建议把交接点往该臂方向挪 2~3cm 后重录。")

        own = own_pose(cfg, arm)
        if float(np.abs(q - own).max()) < 2.0:
            warnings.append(
                f"[{arm}] 交叉位 '{key}' 与自身 table_pick 几乎重合；"
                f"请确认两条臂的两个前方交接点确实是分开的。")

        recorded = (cfg["arms"][arm].get("pose_calib") or {}).get(key)
        if recorded is None:
            warnings.append(
                f"[{arm}] 交叉位 '{key}' 没有标定指纹，无法确认属于当前电机标定。")

    return problems, warnings


def read_firmware(ports: dict[str, str], arms, logger,
                  quiet: bool = False) -> dict[str, dict]:
    """只读电机固件标定（不上力、不写参数）。"""
    firmware: dict[str, dict] = {}
    for arm in arms:
        bus = None
        try:
            bus = FeetechMotorsBus(
                port=ports[arm], motors=make_motors(C.ARM_PREFIX[arm]))
            bus.connect(handshake=False)
            firmware[arm] = calibration_from_bus(bus)
            if not quiet:
                logger.info("[%s] 固件标定读取成功（%s）", arm, ports[arm])
        except Exception as exc:
            logger.warning("[%s] 读固件标定失败: %s", arm, exc)
        finally:
            if bus is not None:
                try:
                    bus.disconnect(disable_torque=False)
                except Exception:
                    pass
    return firmware


def fingerprint_report(cfg: dict, firmware: dict[str, dict],
                       include_cross: bool = True) -> tuple[list[str], list[str]]:
    """比对 10 个标准点 + 交叉点与当前固件标定。"""
    problems: list[str] = []
    warnings: list[str] = []
    for arm, fw in firmware.items():
        missing = C.missing_poses(cfg, arm)
        differs, unknown = C.pose_calib_mismatch(cfg, arm, fw)
        differs = [k for k in differs if k not in missing]
        unknown = [k for k in unknown if k not in missing]
        if differs:
            problems.append(
                f"[{arm}] 这些点位是在旧电机标定下录的，不能直接使用: "
                f"{', '.join(differs)}\n"
                f"  请重跑: python ../2arm/calibrate2arm.py --arm {arm}")
        if unknown:
            warnings.append(
                f"[{arm}] 这些点位没有标定指纹，无法确认是否属于当前电机标定: "
                f"{', '.join(unknown)}")
        if include_cross:
            key = CROSS_KEY[arm]
            q = (cfg["arms"][arm].get("poses") or {}).get(key)
            if q is not None:
                recorded = (cfg["arms"][arm].get("pose_calib") or {}).get(key)
                now = C.calib_fingerprint_short(fw)
                if recorded is None:
                    problems.append(
                        f"[{arm}] 交叉位 '{key}' 没有标定指纹，无法确认属于当前"
                        f"电机标定；请重新运行: "
                        f"python calibrate_handoff.py --arm {arm}")
                elif recorded != now:
                    problems.append(
                        f"[{arm}] 交叉位 '{key}' 是在旧电机标定下录的，"
                        f"请重新运行: python calibrate_handoff.py --arm {arm}")
        warnings.extend(C.tight_pose_report(cfg, arm, fw))
    return problems, warnings


def read_ports(cfg: dict, logger) -> dict[str, str]:
    return resolve_arm_ports(cfg, logger)


class Mover:
    """轨迹封装：多路点移动、双臂并行、速度/温度倍率。"""

    def __init__(self, cfg: dict, dual: DualArm, logger, stop_event):
        self.cfg = cfg
        self.dual = dual
        self.log = logger
        self.stop_event = stop_event
        self.base_speed = 1.0     # 用户 --speed
        self.speed_factor = 1.0   # 温度降速

    def _pose(self, arm: str, item) -> np.ndarray:
        if isinstance(item, str):
            return C.get_pose(self.cfg, arm, item)
        return np.asarray(item, dtype=float)

    def _waypoints(self, arm: str, items) -> list[np.ndarray]:
        return [self._pose(arm, x) for x in items]

    def _trim(self, arm: str, wps: list[np.ndarray]) -> list[np.ndarray]:
        """已经站在某个路点上时从该点开始，避免先退回去再走回来。"""
        if len(wps) <= 1:
            return wps
        a = self.dual.arms[arm]
        ref = getattr(a, "_planned_q", None)
        if ref is None:
            ref = a.hold_q
        if ref is None:
            return wps
        start_k = 0
        for k in range(len(wps)):
            if float(np.abs(wps[k] - ref).max()) <= 1.5:
                start_k = k
        return wps[start_k:] if start_k else wps

    def _sleep(self, seconds: float) -> None:
        end = time.perf_counter() + max(0.0, float(seconds))
        while time.perf_counter() < end:
            self.check_stop()
            time.sleep(min(0.05, end - time.perf_counter()))

    def check_stop(self) -> None:
        if self.stop_event.is_set():
            raise C.StopRequested()

    def mv(self, arm: str, items, *, gripper: Optional[float] = None,
           label: str = "", speed: float = 1.0, precision: bool = False,
           arrive_tol: Optional[float] = None, timeout: Optional[float] = None) -> bool:
        wps = self._trim(arm, self._waypoints(arm, items))
        a = self.dual.arms[arm]
        est = a.start(wps, gripper=gripper, label=f"{arm} {label}",
                      speed_scale=float(speed) * self.base_speed * self.speed_factor,
                      precision=precision)
        tmo = float(timeout) if timeout is not None else est + 12.0
        ok = a.wait(timeout=tmo, tol=arrive_tol)
        if not ok:
            msg = (f"{arm} {label} 未在 {tmo:.1f}s 内到位"
                   f"（容差 {arrive_tol if arrive_tol is not None else self.cfg['motion']['arrive_tol']}）")
            if precision:
                raise C.PhaseError(msg)
            self.log.warning("%s，继续", msg)
        self.check_stop()
        return ok

    def mv_both(self, moves: dict[str, dict]) -> None:
        """两条臂并行移动；用于回 home / 预位，不要求严格到位。"""
        started = {}
        for arm, spec in moves.items():
            try:
                wps = self._trim(arm, self._waypoints(arm, spec["items"]))
                a = self.dual.arms[arm]
                a.start(wps, gripper=spec.get("gripper"),
                        label=f"{arm} {spec.get('label', '')}",
                        speed_scale=(float(spec.get("speed", 1.0))
                                     * self.base_speed * self.speed_factor),
                        precision=bool(spec.get("precision", False)))
                started[arm] = a
            except Exception as exc:
                self.log.warning("[%s] 并行移动启动失败: %s", arm, exc)
        for arm, a in started.items():
            try:
                ok = a.wait(timeout=12.0, tol=moves[arm].get("arrive_tol"))
                if not ok:
                    self.log.warning("[%s] 并行移动未到位，继续", arm)
            except Exception as exc:
                self.log.warning("[%s] 并行移动等待失败: %s", arm, exc)
        self.check_stop()


@dataclass
class ArmGrasp:
    air_closed: Optional[float] = None
    load_available: bool = False
    load_thresh: float = 0.0


class GraspVerifier:
    """空夹基线 + 分步合爪 + 位置/负载双重验证。"""

    def __init__(self, cfg: dict, dual: DualArm, mover: Mover, logger):
        self.cfg = cfg
        self.dual = dual
        self.mover = mover
        self.log = logger
        self.open_g = {a: float(cfg["arms"][a]["gripper_open"]) for a in C.ARMS}
        self.close_g = {a: float(cfg["arms"][a]["gripper_closed"]) for a in C.ARMS}
        self.state = {a: ArmGrasp() for a in C.ARMS}
        self.grasp = cfg["grasp"]

    def measure_air_closed(self, arms=("left", "right")) -> None:
        for arm in arms:
            a = self.dual.arms[arm]
            st = self.state[arm]
            try:
                est = a.start([C.get_pose(self.cfg, arm, "measure")],
                              gripper=self.open_g[arm],
                              label=f"{arm} 去 measure", speed_scale=0.6)
                a.wait(timeout=est + 8.0)
                pos = a.set_gripper(
                    self.close_g[arm],
                    settle=float(self.cfg["motion"]["gripper_close_settle_s"]) + 0.2)
                load = a.read_load()
                time.sleep(float(self.grasp.get("load_settle_s", 0.25)))
                st.air_closed = float(pos)
                st.load_available = 0.5 < float(load) < 300.0
                st.load_thresh = float(load) + float(self.grasp["load_min"])
                a.set_gripper(self.open_g[arm],
                              settle=float(self.cfg["motion"]["gripper_open_settle_s"]))
                self.log.info("[%s] 空夹 pos=%.2f load=%.1f 负载可用=%s 阈值=%.1f",
                              arm, pos, load, st.load_available, st.load_thresh)
                if pos > 20.0:
                    self.log.warning("[%s] 空夹读数偏高（%.1f），measure 位可能被挡",
                                     arm, pos)
            except C.StopRequested:
                raise
            except Exception as exc:
                st.air_closed = self.close_g[arm]
                st.load_available = False
                st.load_thresh = float(self.grasp["load_min"])
                self.log.error("[%s] 测空夹失败: %s", arm, exc)

    def close_and_verify(self, arm: str) -> bool:
        a = self.dual.arms[arm]
        st = self.state[arm]
        target = self.close_g[arm]
        cur = float(a.read_gripper())
        step = float(self.grasp.get("soft_close_step", 8.0))
        while cur > target + 0.5:
            cur = max(target, cur - step)
            a.set_gripper(cur, settle=float(
                self.grasp.get("soft_close_settle_s", 0.10)))
        pos = a.set_gripper(target, settle=float(
            self.cfg["motion"]["gripper_close_settle_s"]))
        time.sleep(float(self.grasp.get("grip_hold_s", 0.15)))
        load = a.read_load()
        time.sleep(float(self.grasp.get("load_settle_s", 0.25)))
        load = min(load, a.read_load())
        base = (st.air_closed if st.air_closed is not None
                else float(st.load_thresh))
        by_pos = pos > base + float(self.grasp["pos_margin"])
        by_load = st.load_available and load >= st.load_thresh
        ok = by_pos or by_load
        self.log.info("[%s] 闭爪: pos=%.2f (空夹 %.2f) load=%.1f -> %s",
                      arm, pos, base, load, "夹住" if ok else "空爪")
        return ok

    def lift_verify(self, arm: str) -> bool:
        a = self.dual.arms[arm]
        st = self.state[arm]
        pos = a.read_gripper()
        load = a.read_load()
        base = (st.air_closed if st.air_closed is not None
                else float(st.load_thresh))
        by_pos = pos > base + float(self.grasp["pos_margin"])
        by_load = st.load_available and load >= st.load_thresh
        ok = by_pos or by_load
        self.log.info("[%s] 抬升后: pos=%.2f load=%.1f -> %s",
                      arm, pos, load, "夹住" if ok else "滑脱")
        return ok

    def open(self, arm: str, settle: Optional[float] = None) -> None:
        a = self.dual.arms[arm]
        a.set_gripper(self.open_g[arm],
                      settle=(float(settle) if settle is not None
                              else float(self.cfg["motion"]["gripper_open_settle_s"])))


class HandoffRunner:
    """手递动作原语：在任意交接点抓起/放置。"""

    def __init__(self, cfg: dict, dual: DualArm, logger, stop_event,
                 pick_tries: int = 3, overlap: bool = True):
        self.cfg = cfg
        self.dual = dual
        self.log = logger
        self.stop_event = stop_event
        self.mover = Mover(cfg, dual, logger, stop_event)
        self.grasp = GraspVerifier(cfg, dual, self.mover, logger)
        self.open_g = self.grasp.open_g
        self.close_g = self.grasp.close_g
        self.pick_tries = max(1, int(pick_tries))
        self.overlap = bool(overlap)
        self.lift_speed = float(cfg["grasp"].get("lift_speed", 0.3))
        self.release_dwell = float(cfg["motion"].get("release_dwell_s", 0.18))

    # ---- 位姿 ----
    def spot_pose(self, arm: str, front: str) -> np.ndarray:
        return spot_pose(self.cfg, arm, front)

    # ---- 抓取 ----
    def pick_at(self, arm: str, front: str, tries: Optional[int] = None) -> bool:
        """到示教点抓取。所有目标点都原样使用示教位姿，不做任何偏移。

        own 点：carry -> table_approach -> table_hover -> table_pick
        cross 点：carry -> place_right/place_left（该点本身就是抓取高度）
        重试也是回到同一个示教点，不换 pan/深度。
        """
        own = (front == arm)
        base = self.spot_pose(arm, front)
        n = self.pick_tries if tries is None else max(1, int(tries))
        self.log.info("[%s] 在 %s 前方抓取（示教点 %s，最多 %d 次）",
                      arm, front, C.fmt_q(base), n)
        if own:
            self.mover.mv(arm, ["carry", "table_approach", "table_hover"],
                          gripper=self.open_g[arm],
                          label=f"去 {front} 观察位", speed=1.0)
            lift_items = ["table_hover"]
        else:
            self.mover.mv(arm, ["carry"], gripper=self.open_g[arm],
                          label="去搬运位", speed=1.0)
            lift_items = []

        for i in range(n):
            self.mover.mv(arm, [base], gripper=self.open_g[arm],
                          label=f"到示教抓取点 {i + 1}", speed=0.75,
                          precision=True, arrive_tol=5.0)
            if self.grasp.close_and_verify(arm):
                if lift_items:
                    self.mover.mv(arm, lift_items, gripper=self.close_g[arm],
                                  label="按示教点抬起", speed=self.lift_speed)
                else:
                    self.mover.mv(arm, ["carry"], gripper=self.close_g[arm],
                                  label="离开示教点", speed=self.lift_speed)
                if self.grasp.lift_verify(arm):
                    if lift_items:
                        self.mover.mv(arm, ["carry"], gripper=self.close_g[arm],
                                      label="到搬运位", speed=1.0)
                    return True
                self.log.warning("[%s] 抬起后滑脱，重试同一示教点", arm)
            self.grasp.open(arm)
            if i + 1 < n:
                self.mover.mv(arm, ["table_hover" if own else "carry"],
                              gripper=self.open_g[arm],
                              label="回退位", speed=1.0)
        self.grasp.open(arm)
        self.mover.mv(arm, ["carry"], gripper=self.open_g[arm],
                      label="抓取失败退出", speed=1.0)
        return False

    # ---- 放置 ----
    def place_at(self, arm: str, front: str) -> bool:
        """放到示教点。释放点就是示教位姿本身，不抬高、不做释放偏移。

        own 点：carry -> table_approach -> table_hover -> table_pick
        cross 点：carry -> place_right/place_left
        放完按原路退回（own 经 table_hover/table_approach，cross 直接回 carry）。
        """
        own = (front == arm)
        base = self.spot_pose(arm, front)
        self.log.info("[%s] 放到 %s 前方（示教点 %s）", arm, front, C.fmt_q(base))
        if own:
            self.mover.mv(arm, ["carry", "table_approach", "table_hover"],
                          gripper=self.close_g[arm],
                          label=f"去 {front} 释放位", speed=1.0)
            retreat = ["table_hover", "table_approach", "carry"]
        else:
            self.mover.mv(arm, ["carry"], gripper=self.close_g[arm],
                          label="去搬运位", speed=1.0)
            retreat = ["carry"]
        # 下探失败会让方块留在夹爪里；让异常冒出去，状态机据此复检是否还夹住。
        self.mover.mv(arm, [base], gripper=self.close_g[arm],
                      label="到示教释放点", speed=0.85,
                      precision=True, arrive_tol=5.0)
        self.grasp.open(arm, settle=self.release_dwell + 0.1)
        try:
            self.mover.mv(arm, retreat, gripper=self.open_g[arm],
                          label="按示教点退回", speed=0.9)
        except C.StopRequested:
            raise
        except Exception as exc:
            # 已经开爪，方块已经在示教点上；退回失败只记录，不推翻放置结果。
            self.log.warning("[%s] 放置后退回未完成: %s", arm, exc)
        try:
            load = self.dual.arms[arm].read_load()
            pos = self.dual.arms[arm].read_gripper()
            self.log.info("[%s] 释放后 pos=%.1f load=%.1f", arm, pos, load)
        except Exception:
            pass
        return True

    # ---- 过渡/回位 ----
    def go_home(self, arm: str) -> None:
        self.mover.mv(arm, ["carry", "home"], gripper=self.open_g[arm],
                      label="回 home", speed=1.0)

    def home_both(self) -> None:
        self.mover.mv_both({
            a: {"items": ["carry", "home"], "gripper": self.open_g[a],
                "label": "回 home", "speed": 1.0}
            for a in C.ARMS
        })

    def safe_retreat(self, arm: str) -> None:
        try:
            self.grasp.open(arm)
        except Exception:
            pass
        try:
            self.go_home(arm)
        except Exception as exc:
            self.log.warning("[%s] 安全退出失败: %s", arm, exc)

    def transition_after_place(self, place_arm: str, target_front: str) -> None:
        """放完后：放置臂回 home，下一条臂预位到自己的 table_approach。"""
        next_arm = target_front
        if next_arm == place_arm:
            self.go_home(place_arm)
            return
        if not self.overlap:
            self.go_home(place_arm)
            self.mover.mv(next_arm, ["carry", "table_approach"],
                          gripper=self.open_g[next_arm],
                          label="预位到抓取上方", speed=0.9)
            return
        self.mover.mv_both({
            place_arm: {"items": ["carry", "home"],
                        "gripper": self.open_g[place_arm],
                        "label": "回 home", "speed": 1.0},
            next_arm: {"items": ["carry", "table_approach"],
                       "gripper": self.open_g[next_arm],
                       "label": "预位到抓取上方", "speed": 0.9},
        })

    # ---- 保护 ----
    def check_temp(self, allow_cool: bool = True) -> float:
        try:
            temp = float(self.dual.max_temp())
        except Exception as exc:
            self.log.debug("读温度失败: %s", exc)
            return 1.0
        stop = float(self.cfg["safety"]["temp_stop_c"])
        slow = float(self.cfg["safety"]["temp_slow_c"])
        warn = float(self.cfg["safety"]["temp_warn_c"])
        self.log.info("当前最高电机温度 %.1f°C", temp)
        if temp >= stop:
            if not allow_cool:
                self.log.warning("温度 %.1f°C 达到停机阈值，但方块还在夹爪里，"
                                 "先完成本次放置再冷却", temp)
                return 1.0
            self.log.error("温度 %.1f°C 达到停机阈值，卸力冷却 %.0fs",
                           temp, float(self.cfg["safety"]["temp_wait_s"]))
            self.dual.disable_torque()
            self.mover._sleep(float(self.cfg["safety"]["temp_wait_s"]))
            self.dual.enable_torque()
            self.home_both()
            self.grasp.measure_air_closed()
            self.mover.speed_factor = 1.0
            return 1.0
        if temp >= slow:
            self.mover.speed_factor = float(
                self.cfg["motion"].get("temp_slow_scale", 0.7))
            self.log.warning("温度 %.1f°C >= %.0f°C，降速到 %.0f%%",
                             temp, slow, self.mover.speed_factor * 100)
            return self.mover.speed_factor
        if temp >= warn:
            self.log.warning("温度 %.1f°C 偏高（警告阈值 %.0f°C）", temp, warn)
        self.mover.speed_factor = 1.0
        return 1.0

    def ensure_bus(self, arms=("left", "right")) -> None:
        threshold = int(self.cfg["safety"].get("reconnect_after_errors", 6))
        for arm in arms:
            try:
                streak = int(self.dual.arms[arm].bus.error_streak)
            except Exception:
                continue
            if streak >= threshold:
                self.log.warning("[%s] 总线连续错误 %d 次，重连", arm, streak)
                self.dual.reconnect(arm)

class HandoffState:
    """方块在两个交接点之间来回；失败时四点搜索恢复。"""

    def __init__(self, runner: HandoffRunner, logger, pick_tries: int = 5):
        self.r = runner
        self.log = logger
        self.pick_tries = int(pick_tries)
        self.cube_front = "left"      # 方块所在的交接点
        self.cube_hand: Optional[str] = None
        self.last_front = "left"      # 抓住它时所在的交接点
        self.unknown = False

    def step(self) -> dict:
        try:
            if self.unknown:
                return self._recover_unknown()
            if self.cube_hand is None:
                return self._do_pick()
            return self._do_place()
        except C.StopRequested:
            raise
        except Exception as exc:
            self.log.error("手递步骤失败: %s: %s", type(exc).__name__, exc)
            self.log.debug(traceback.format_exc())
            return self._fail(exc)

    def _do_pick(self) -> dict:
        front = self.cube_front
        for arm in (front, opposite(front)):
            self.log.info("[%s] 在 %s 前方抓起", arm, front)
            if self.r.pick_at(arm, front, tries=self.pick_tries):
                self.cube_hand = arm
                self.last_front = front
                return {"status": "picked", "arm": arm, "front": front}
        self.unknown = True
        return {"status": "pick_failed", "front": front}

    def _do_place(self) -> dict:
        arm = self.cube_hand
        target = opposite(self.last_front)
        self.log.info("[%s] 把方块放到 %s 前方", arm, target)
        try:
            self.r.place_at(arm, target)
        except C.StopRequested:
            raise
        except Exception as exc:
            try:
                holding = self.r.grasp.lift_verify(arm)
            except Exception:
                holding = False
            if holding:
                self.log.warning("[%s] 放置失败但仍夹住，下一步重试", arm)
                return {"status": "place_retry", "arm": arm, "error": str(exc)}
            self.cube_hand = None
            self.unknown = True
            return {"status": "lost", "arm": arm, "error": str(exc)}
        self.cube_front = target
        self.cube_hand = None
        try:
            self.r.transition_after_place(arm, target)
        except C.StopRequested:
            raise
        except Exception as exc:
            self.log.warning("放置后过渡失败: %s", exc)
        return {"status": "placed", "arm": arm, "front": target,
                "cycle": target == "left"}

    def _fail(self, exc) -> dict:
        arm = self.cube_hand
        if arm is not None:
            try:
                holding = self.r.grasp.lift_verify(arm)
            except Exception:
                holding = False
            if holding:
                return {"status": "place_retry", "arm": arm, "error": str(exc)}
            self.cube_hand = None
        self.unknown = True
        try:
            self.r.home_both()
        except Exception:
            pass
        return {"status": "error", "error": str(exc)}

    def _recover_unknown(self) -> dict:
        self.log.warning("方块位置未知，开始四点搜索（每个点最多 2 次）")
        for front in ("left", "right"):
            for arm in (front, opposite(front)):
                self.log.info("[%s] 搜索 %s 前方", arm, front)
                if self.r.pick_at(arm, front, tries=2):
                    self.cube_front = front
                    self.cube_hand = arm
                    self.last_front = front
                    self.unknown = False
                    return {"status": "recovered", "arm": arm, "front": front}
        try:
            self.r.home_both()
        except Exception:
            pass
        return {"status": "not_found"}


def shutdown(dual: Optional[DualArm], logger) -> None:
    """统一退出：停线程 -> 按当前位置开爪 -> 卸力 -> 断开。"""
    if dual is None:
        return
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except Exception:
        pass
    try:
        dual.stop_workers()
    except Exception as exc:
        logger.debug("停止轨迹线程失败: %s", exc)
    for arm in C.ARMS:
        try:
            q = dual.arms[arm].read_joints()
            value = float(dual.cfg["arms"][arm]["gripper_open"])
            dual.arms[arm].bus.write_goal(q, value)
        except Exception as exc:
            logger.debug("[%s] 退出开爪失败: %s", arm, exc)
    try:
        dual.disable_torque()
    except Exception as exc:
        logger.debug("退出卸力失败: %s", exc)
    try:
        dual.disconnect()
    except Exception as exc:
        logger.debug("退出断开失败: %s", exc)
    logger.info("退出清理完成（已连接的臂会开爪、卸力并断开）。")
