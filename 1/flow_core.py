#!/usr/bin/env python
"""flow_core — 10 步示教手递动作流（预设、校验、原样回放）。

每条臂的动作流固定为 10 步，顺序和夹爪动作都是预设的；示教只负责
每条臂在每一步的 5 个关节角度。运行主程序时把 10 步的 q 和夹爪目标
原样发送，不做任何偏移/抬高/派生位姿。

左臂 10 步（右臂把 place_right 换成 place_left）：
  1 home            夹爪 open
  2 carry           open
  3 table_approach  open
  4 table_hover     open
  5 table_pick      open  -> 到位后合爪
  6 table_hover     closed
  7 carry           closed
  8 place_right     closed -> 到位后开爪
  9 carry           open
 10 home            open

flow JSON 里每步字段：
  q         5 个关节，示教值，原样发送
  gripper   移动到该步时夹爪目标（open/closed 的数值）
  grip_after 到位后的夹爪动作，null 表示不动
  dwell     到位后停留秒数（不改变位置）
  precision 该步是否按精下探速度执行
  label     预设动作名，仅用于阅读/校验
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import armcore as C
from handoff_core import GraspVerifier, Mover, pose_margins

N_STEPS = 10

# (label, 移动时夹爪, 到位后动作, 到位后停留秒, 是否精下探)
FLOW_TEMPLATE = {
    "left": [
        ("home", "open", None, 0.10, False),
        ("carry", "open", None, 0.10, False),
        ("table_approach", "open", None, 0.10, False),
        ("table_hover", "open", None, 0.10, False),
        ("table_pick", "open", "closed", 0.10, True),
        ("table_hover", "closed", None, 0.10, False),
        ("carry", "closed", None, 0.10, False),
        ("place_right", "closed", "open", 0.10, False),
        ("carry", "open", None, 0.10, False),
        ("home", "open", None, 0.10, False),
    ],
    "right": [
        ("home", "open", None, 0.10, False),
        ("carry", "open", None, 0.10, False),
        ("table_approach", "open", None, 0.10, False),
        ("table_hover", "open", None, 0.10, False),
        ("table_pick", "open", "closed", 0.10, True),
        ("table_hover", "closed", None, 0.10, False),
        ("carry", "closed", None, 0.10, False),
        ("place_left", "closed", "open", 0.10, False),
        ("carry", "open", None, 0.10, False),
        ("home", "open", None, 0.10, False),
    ],
}


def flow_path(flow_dir: Path, arm: str) -> Path:
    return Path(flow_dir) / f"{arm}_handoff_10.json"


def _gripper(cfg: dict, arm: str, which: str) -> float:
    value = cfg["arms"][arm].get(f"gripper_{which}")
    if value is None:
        raise SystemExit(
            f"[{arm}] 缺少 gripper_{which}，无法生成动作流。\n"
            f"请先运行: python ../2arm/calibrate2arm.py --arm {arm}")
    return float(value)


def build_flow(cfg: dict, arm: str, require_cross: bool = True) -> dict:
    """从 config.json 的示教位姿生成预设 10 步流。"""
    steps = []
    for i, (label, before, after, dwell, precision) in enumerate(
            FLOW_TEMPLATE[arm], 1):
        try:
            q = C.get_pose(cfg, arm, label)
        except SystemExit:
            if label in ("place_right", "place_left"):
                if require_cross:
                    raise
                # 还没有交叉位时，用 carry 作为示教起点，让人工把第 8 步摆到对方前方。
                q = C.get_pose(cfg, arm, "carry")
            else:
                raise
        steps.append({
            "index": i,
            "label": label,
            "q": [round(float(v), 4) for v in np.asarray(q, dtype=float)],
            "gripper": _gripper(cfg, arm, before),
            "grip_after": _gripper(cfg, arm, after) if after else None,
            "dwell": float(dwell),
            "precision": bool(precision),
            "source": "config",
        })

    return {
        "version": 1,
        "name": f"{arm}_handoff_10",
        "arm": arm,
        "fingerprint": (cfg["arms"][arm].get("pose_calib") or {}).get("home"),
        "created": datetime.now().isoformat(timespec="seconds"),
        "steps": steps,
    }


def save_flow(path: Path, flow: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(flow, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def load_flow(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"找不到动作流文件: {path}\n"
            f"请先生成预设: python teach_handoff_flow.py --seed")
    try:
        flow = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"动作流解析失败 {path}: {exc}")
    return flow


def validate_flow(flow: dict, arm: str,
                  firmware: dict | None = None) -> tuple[list[str], list[str]]:
    """返回 (阻断问题, 警告)。"""
    problems: list[str] = []
    warnings: list[str] = []

    if flow.get("arm") != arm:
        problems.append(f"flow.arm={flow.get('arm')!r} 与文件名对应的 {arm!r} 不一致")
    steps = flow.get("steps") or []
    if len(steps) != N_STEPS:
        problems.append(f"动作流应有 {N_STEPS} 步，实际 {len(steps)} 步")

    for i, step in enumerate(steps, 1):
        q = np.asarray(step.get("q") or [], dtype=float)
        if q.shape != (C.NJ,):
            problems.append(f"第 {i} 步 q 需要 {C.NJ} 个数，实际 {q.shape}")
            continue
        if not np.all(np.isfinite(q)):
            problems.append(f"第 {i} 步 q 含 NaN/Inf")
        if float(np.abs(q).max()) >= 100.0 - 1e-3:
            bad = C.saturated_joints(q)
            problems.append(f"第 {i} 步 q 顶到 ±100（{', '.join(bad)}），不可用")
        tight = [f"{n}({m:.1f})" for n, m in pose_margins(q) if m < 1.0]
        near = [f"{n}({m:.1f})" for n, m in pose_margins(q) if 1.0 <= m < 2.0]
        if tight:
            warnings.append(f"第 {i} 步离边界很近: {', '.join(tight)}")
        elif near:
            warnings.append(f"第 {i} 步接近边界: {', '.join(near)}")

        for field in ("gripper", "grip_after"):
            value = step.get(field)
            if value is None and field == "grip_after":
                continue
            try:
                value = float(value)
            except Exception:
                problems.append(f"第 {i} 步 {field} 不是数值: {step.get(field)!r}")
                continue
            if not (0.0 <= value <= 100.0):
                problems.append(f"第 {i} 步 {field}={value} 超出 0..100")
        if float(step.get("dwell", 0.0)) < 0.0:
            problems.append(f"第 {i} 步 dwell 不能为负")

        if i <= len(FLOW_TEMPLATE[arm]):
            want = FLOW_TEMPLATE[arm][i - 1][0]
            got = step.get("label")
            if got != want:
                warnings.append(f"第 {i} 步 label={got!r}，预设顺序是 {want!r}")

    fp = flow.get("fingerprint")
    if not fp:
        problems.append("动作流没有电机标定指纹，无法确认属于当前硬件")
    elif firmware is not None:
        now = C.calib_fingerprint_short(firmware)
        if fp != now:
            problems.append(
                f"[{arm}] 动作流是在旧电机标定下示教的（{fp} != {now}）；"
                f"请重新运行 teach_handoff_flow.py --arm {arm}")

    return problems, warnings


def print_flow(flow: dict) -> None:
    arm = flow.get("arm")
    print(f"\n[{arm}] {flow.get('name')}  fingerprint={flow.get('fingerprint')}")
    print(f"  {'#':>2}  {'label':<16} {'夹爪':>5} {'到位后':>6} "
          f"{'dwell':>5} {'精':>2}  q")
    for step in flow.get("steps") or []:
        after = step.get("grip_after")
        after_s = "-" if after is None else f"{float(after):.1f}"
        q = C.fmt_q(np.asarray(step.get("q") or [], dtype=float))
        print(f"  {step.get('index'):>2}  {str(step.get('label')):<16} "
              f"{float(step.get('gripper', 0)):>5.1f} {after_s:>6} "
              f"{float(step.get('dwell', 0)):>5.2f} "
              f"{'Y' if step.get('precision') else '-':>2}  {q}")


class FlowRunner:
    """按示教 flow 的 10 步原样执行。"""

    def __init__(self, cfg: dict, dual, logger, stop_event, speed: float = 1.0):
        self.cfg = cfg
        self.dual = dual
        self.log = logger
        self.stop_event = stop_event
        self.mover = Mover(cfg, dual, logger, stop_event)
        self.grasp = GraspVerifier(cfg, dual, self.mover, logger)
        self.open_g = self.grasp.open_g
        self.close_g = self.grasp.close_g
        self.mover.base_speed = float(speed)
        self.close_settle = float(cfg["motion"]["gripper_close_settle_s"])
        self.open_settle = float(cfg["motion"]["gripper_open_settle_s"])
        self.holding = {a: False for a in C.ARMS}

    def run_flow(self, flow: dict) -> bool:
        try:
            return self._run_flow(flow)
        except C.StopRequested:
            raise
        except Exception as exc:
            self.log.error("[%s] 动作流执行失败: %s: %s",
                           flow.get("arm"), type(exc).__name__, exc)
            return False

    def _run_flow(self, flow: dict) -> bool:
        arm = flow["arm"]
        steps = flow["steps"]
        self.log.info("=" * 60)
        self.log.info("[%s] 开始 10 步示教动作流", arm)
        for i, step in enumerate(steps, 1):
            self.mover.check_stop()
            q = np.asarray(step["q"], dtype=float)
            ok = self.mover.mv(
                arm, [q], gripper=float(step["gripper"]),
                label=f"第 {i}/{len(steps)} 步 {step.get('label', '')}",
                speed=1.0, precision=bool(step.get("precision", False)),
                arrive_tol=5.0)
            if not ok:
                self.log.warning("[%s] 第 %d 步未到位，动作流中止", arm, i)
                return False
            dwell = float(step.get("dwell", 0.0))
            if dwell > 0:
                self.mover._sleep(dwell)

            after = step.get("grip_after")
            if after is not None:
                value = float(after)
                settle = (self.close_settle
                          if abs(value - self.close_g[arm]) <= abs(value - self.open_g[arm])
                          else self.open_settle)
                self.dual.arms[arm].set_gripper(value, settle=settle)
                self.holding[arm] = (
                    abs(value - self.close_g[arm]) < abs(value - self.open_g[arm]))
                self.mover._sleep(0.10)
                try:
                    pos = self.dual.arms[arm].read_gripper()
                    load = self.dual.arms[arm].read_load()
                    self.log.info("[%s] 第 %d 步夹爪 -> %.1f（pos=%.1f load=%.1f）",
                                  arm, i, value, pos, load)
                except Exception:
                    pass
            self.log.info("[%s] 第 %d/%d 步完成: %s", arm, i, len(steps),
                          step.get("label", ""))
        self.log.info("[%s] 10 步动作流完成", arm)
        return True

    def home_both(self, flows: dict | None = None) -> None:
        moves = {}
        for arm in C.ARMS:
            flow = (flows or {}).get(arm)
            if flow and len(flow.get("steps") or []) == N_STEPS:
                q = np.asarray(flow["steps"][-1]["q"], dtype=float)
            else:
                q = C.get_pose(self.cfg, arm, "home")
            moves[arm] = {"items": [q], "gripper": self.open_g[arm],
                          "label": "回 home", "speed": 1.0}
        self.mover.mv_both(moves)

    def has_holding(self) -> bool:
        return any(bool(v) for v in self.holding.values())

    def safe_home(self, flows: dict | None = None) -> None:
        held = [a for a in C.ARMS if self.holding.get(a)]
        if held:
            self.log.warning("仍夹着方块（%s），不自动回 home；停在当前位置",
                             ", ".join(held))
            return
        try:
            self.home_both(flows)
        except Exception as exc:
            self.log.warning("回 home 未完成: %s", exc)

    def check_temp(self, flows: dict | None = None) -> float:
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
            self.log.error("温度 %.1f°C 达到停机阈值，卸力冷却 %.0fs",
                           temp, float(self.cfg["safety"]["temp_wait_s"]))
            self.dual.disable_torque()
            self.mover._sleep(float(self.cfg["safety"]["temp_wait_s"]))
            self.dual.enable_torque()
            self.safe_home(flows)
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

    def ensure_bus(self) -> None:
        threshold = int(self.cfg["safety"].get("reconnect_after_errors", 6))
        for arm in C.ARMS:
            try:
                streak = int(self.dual.arms[arm].bus.error_streak)
            except Exception:
                continue
            if streak >= threshold:
                self.log.warning("[%s] 总线连续错误 %d 次，重连", arm, streak)
                self.dual.reconnect(arm)
