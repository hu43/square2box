#!/usr/bin/env python
"""taught_flow_core — 14 步预设动作流的定义、校验与读写。

预设顺序（严格固定）：

    左臂：安全位 -> 夹方块前 -> 夹方块 -> 夹起在中转位
          -> 放方块前 -> 放方块 -> 回安全位
    右臂：安全位 -> 夹方块前 -> 夹方块 -> 夹起在中转位
          -> 放方块前 -> 放方块 -> 回安全位

示教文件 taught_flow.json 只保存每一步的 5 个关节值 + 录制时的电机
标定指纹。运行时按索引 1..14 顺序发送这些关节值，不改位置、不插入
偏移点；夹爪在每个预设步骤按固定状态开/合。

夹爪状态约定：
  pre  = 到达该步骤前夹爪应处于的状态
  post = 到达该步骤后要执行的开/合动作（None 表示不变）
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import armcore as C
from handoff_core import Mover, pose_margins

HERE = Path(__file__).resolve().parent
FLOW_PATH = HERE / "taught_flow.json"
FLOW_VERSION = 1

STEP_DEFS: list[dict] = [
    {"index": 1, "arm": "left", "name": "left_safe",
     "label": "左臂安全位", "pre": "open", "post": None},
    {"index": 2, "arm": "left", "name": "left_pick_pre",
     "label": "左臂夹方块前", "pre": "open", "post": None},
    {"index": 3, "arm": "left", "name": "left_pick",
     "label": "左臂夹方块", "pre": "open", "post": "closed"},
    {"index": 4, "arm": "left", "name": "left_carry",
     "label": "左臂夹起方块并在中转位", "pre": "closed", "post": None},
    {"index": 5, "arm": "left", "name": "left_place_pre",
     "label": "左臂放方块前", "pre": "closed", "post": None},
    {"index": 6, "arm": "left", "name": "left_place",
     "label": "左臂放方块", "pre": "closed", "post": "open"},
    {"index": 7, "arm": "left", "name": "left_return",
     "label": "左臂回安全位", "pre": "open", "post": None},
    {"index": 8, "arm": "right", "name": "right_safe",
     "label": "右臂安全位", "pre": "open", "post": None},
    {"index": 9, "arm": "right", "name": "right_pick_pre",
     "label": "右臂夹方块前", "pre": "open", "post": None},
    {"index": 10, "arm": "right", "name": "right_pick",
     "label": "右臂夹方块", "pre": "open", "post": "closed"},
    {"index": 11, "arm": "right", "name": "right_carry",
     "label": "右臂夹起方块并在中转位", "pre": "closed", "post": None},
    {"index": 12, "arm": "right", "name": "right_place_pre",
     "label": "右臂放方块前", "pre": "closed", "post": None},
    {"index": 13, "arm": "right", "name": "right_place",
     "label": "右臂放方块", "pre": "closed", "post": "open"},
    {"index": 14, "arm": "right", "name": "right_return",
     "label": "右臂回安全位", "pre": "open", "post": None},
]


def empty_flow() -> dict:
    return {
        "version": FLOW_VERSION,
        "updated": None,
        "steps": [
            {"index": d["index"], "arm": d["arm"], "name": d["name"],
             "label": d["label"], "pre": d["pre"], "post": d["post"],
             "q": None, "pose_calib": None}
            for d in STEP_DEFS
        ],
    }


def normalize_flow(raw: dict | None) -> dict:
    """以 STEP_DEFS 为准生成完整 14 步骨架，只保留已有 q/指纹。"""
    raw = raw or {}
    old = {}
    for s in raw.get("steps") or []:
        try:
            old[int(s.get("index"))] = s
        except Exception:
            continue
    flow = empty_flow()
    flow["version"] = raw.get("version", FLOW_VERSION)
    flow["updated"] = raw.get("updated")
    for step in flow["steps"]:
        got = old.get(step["index"]) or {}
        q = got.get("q")
        if q is not None:
            try:
                arr = np.asarray(q, dtype=float)
                if arr.shape == (C.NJ,) and np.all(np.isfinite(arr)):
                    step["q"] = [round(float(v), 4) for v in arr]
            except Exception:
                step["q"] = None
        step["pose_calib"] = got.get("pose_calib")
    return flow


def load_flow(path: Path = FLOW_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return empty_flow()
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"动作流文件解析失败 {path}: {exc}")
    return normalize_flow(raw)


def save_flow(flow: dict, path: Path = FLOW_PATH) -> None:
    path = Path(path)
    flow = normalize_flow(flow)
    flow["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(flow, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def gripper_values(cfg: dict, arm: str) -> tuple[float, float]:
    go = cfg["arms"][arm].get("gripper_open")
    gc = cfg["arms"][arm].get("gripper_closed")
    if go is None or gc is None:
        raise SystemExit(
            f"[{arm}] 缺少 gripper_open/gripper_closed，请先运行: "
            f"python ../2arm/calibrate2arm.py --arm {arm}")
    return float(go), float(gc)


def validate_flow(flow: dict, cfg: dict,
                  require_complete: bool = True) -> tuple[list[str], list[str]]:
    """校验预设动作流。返回 (阻断问题, 警告)。"""
    problems: list[str] = []
    warnings: list[str] = []
    flow = normalize_flow(flow)

    for arm in ("left", "right"):
        try:
            gripper_values(cfg, arm)
        except SystemExit as exc:
            problems.append(str(exc))

    qmap = {}
    for d in STEP_DEFS:
        step = flow["steps"][d["index"] - 1]
        q = step.get("q")
        if q is None:
            (problems if require_complete else warnings).append(
                f"第 {d['index']} 步「{d['label']}」还没有示教关节值")
            continue
        arr = np.asarray(q, dtype=float)
        if arr.shape != (C.NJ,):
            problems.append(f"第 {d['index']} 步「{d['label']}」需要 {C.NJ} 个关节值，"
                            f"实际 {arr.shape}")
            continue
        if not np.all(np.isfinite(arr)):
            problems.append(f"第 {d['index']} 步「{d['label']}」含非法数值")
            continue
        if np.any(arr < -100.0) or np.any(arr > 100.0):
            problems.append(f"第 {d['index']} 步「{d['label']}」超出 [-100,100]")
        bad = C.saturated_joints(arr)
        if bad:
            problems.append(
                f"第 {d['index']} 步「{d['label']}」顶到 ±100（{', '.join(bad)}），"
                f"该点不可用；请重录并把交接点往可达方向挪一点。")
        tight = [f"{n}({m:.1f})" for n, m in pose_margins(arr) if m < 2.0]
        if tight:
            warnings.append(
                f"第 {d['index']} 步「{d['label']}」离标定边界很近: {', '.join(tight)}")
        if step.get("pose_calib") is None:
            warnings.append(
                f"第 {d['index']} 步「{d['label']}」没有标定指纹，运行时会被拒绝；"
                f"请重新示教。")
        qmap[d["index"]] = arr

    # 抓取前 -> 抓取 应该接近竖直，提醒但不阻断。
    for pre_i, pick_i in ((2, 3), (9, 10)):
        a, b = qmap.get(pre_i), qmap.get(pick_i)
        if a is None or b is None:
            continue
        d = b - a
        for idx, joint in ((0, "shoulder_pan"), (4, "wrist_roll")):
            if abs(float(d[idx])) > 2.0:
                warnings.append(
                    f"第 {pre_i}->{pick_i} 步 {joint} 变化 {d[idx]:+.1f}，"
                    f"下探不是竖直的；抓取可能偏，但仍会按示教值执行。")

    # 夹爪状态链必须连续：上一步结束状态 == 下一步开始状态
    state = None
    for d in STEP_DEFS:
        step = flow["steps"][d["index"] - 1]
        pre, post = step["pre"], step["post"]
        if state is not None and pre != state:
            problems.append(
                f"第 {d['index']} 步「{d['label']}」夹爪 pre={pre}，"
                f"但上一步结束是 {state}；预设动作流被改坏了。")
        state = post or pre

    return problems, warnings


def flow_fingerprint_report(flow: dict, firmware: dict[str, dict]
                            ) -> tuple[list[str], list[str]]:
    """检查 14 步都录制于当前电机标定。"""
    problems: list[str] = []
    warnings: list[str] = []
    flow = normalize_flow(flow)
    for d in STEP_DEFS:
        step = flow["steps"][d["index"] - 1]
        if step.get("q") is None:
            continue
        arm = d["arm"]
        fw = firmware.get(arm)
        if fw is None:
            problems.append(f"第 {d['index']} 步「{d['label']}」读不到 {arm} 臂固件标定")
            continue
        recorded = step.get("pose_calib")
        now = C.calib_fingerprint_short(fw)
        if recorded is None:
            problems.append(
                f"第 {d['index']} 步「{d['label']}」没有标定指纹；"
                f"请重新运行 calibrate_taught_flow.py")
        elif recorded != now:
            problems.append(
                f"第 {d['index']} 步「{d['label']}」是在旧电机标定下录的；"
                f"请重新运行 calibrate_taught_flow.py")
    return problems, warnings


def set_gripper_only(dual, cfg: dict, arm: str, state: str,
                     settle: float | None = None) -> float:
    """只写夹爪目标，不动 5 个关节。

    示教时关节已卸力，不能走 Arm.set_gripper（它会连带写关节目标，
    把用户手摆的臂拉回旧位置）。
    """
    go, gc = gripper_values(cfg, arm)
    value = go if state == "open" else gc
    a = dual.arms[arm]
    bus = getattr(a.bus, "bus", None)
    if bus is None:
        raise RuntimeError(f"[{arm}] 未连接，无法设置夹爪")
    name = a.bus.gripper_name
    bus.sync_write("Goal_Position", {name: float(value)})
    bus.sync_write("Torque_Enable", {name: 1})
    if settle is None:
        settle = float(cfg["motion"][
            "gripper_open_settle_s" if state == "open"
            else "gripper_close_settle_s"])
    time.sleep(float(settle))
    try:
        return float(a.read_gripper())
    except Exception:
        return float(value)


def set_joint_torque(dual, arm: str, on: bool) -> None:
    """只开/关 5 个关节电机的力矩，夹爪力矩保持不变。

    示教“夹着方块在中转位”时，关节卸力方便用手移动，但夹爪必须继续
    用力夹住方块，否则方块会掉。
    """
    a = dual.arms[arm]
    bus = getattr(a.bus, "bus", None)
    if bus is None:
        raise RuntimeError(f"[{arm}] 未连接，无法设置关节力矩")
    names = a.bus.joint_names
    if on:
        pos = bus.sync_read("Present_Position", names, normalize=False)
        bus.sync_write(
            "Goal_Position",
            {n: float(pos[n]) for n in names}, normalize=False)
        bus.sync_write("Torque_Enable", {n: 1 for n in names})
    else:
        bus.sync_write("Torque_Enable", {n: 0 for n in names})


def step_by_index(flow: dict, index: int) -> dict:
    flow = normalize_flow(flow)
    return flow["steps"][int(index) - 1]


def steps_for_arm(flow: dict, arm: str) -> list[dict]:
    flow = normalize_flow(flow)
    return [s for s in flow["steps"] if s["arm"] == arm]


def step_gripper_text(step: dict) -> str:
    post = step.get("post")
    if post is None:
        return f"夹爪保持 {step['pre']}"
    return f"夹爪 {step['pre']} -> {post}"


def make_mover(cfg: dict, dual, logger, stop_event) -> Mover:
    return Mover(cfg, dual, logger, stop_event)
