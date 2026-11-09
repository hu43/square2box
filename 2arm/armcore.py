#!/usr/bin/env python
"""armcore — 双臂程序的公共层。

包含：
  - 关节/位姿常量与标定项定义
  - config.json 的默认值、深度合并、读写
  - 轨迹规划（Catmull-Rom 路径 + 限速时间参数化 + 端点缓起缓停）

坐标约定（与 lerobot 的 RANGE_M100_100 一致）：
  每个关节一个数，范围 [-100, 100]，单位是归一化角度；
  夹爪是 0..100，0 = 闭合，100 = 张开。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
LOG_DIR = HERE / "logs"
CALIB_DIR = HERE / "calib"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER = "gripper"
NJ = len(JOINTS)
ARMS = ("left", "right")
ARM_PREFIX = {"left": "left_arm", "right": "right_arm"}

# 每条臂需要人工标定的位姿。顺序即标定向导的顺序。
POSE_KEYS = [
    "measure",
    "home",
    "carry",
    "table_approach",
    "table_hover",
    "table_pick",
    "box_approach",
    "box_hover",
    "box_pick",
    "box_drop",
]

POSE_HELP = {
    "measure": (
        "空合测量位",
        "夹爪完全悬空、周围没有任何物体（离桌面、盒子、其它臂都远）。\n"
        "    程序会在这里闭合夹爪，测出“空夹”读数作为抓取判定的基线。",
    ),
    "home": (
        "安全休息位",
        "两臂都不工作时停放的位置。要避开桌面、盒子和另一条臂的活动范围。",
    ),
    "carry": (
        "搬运中转位",
        "高于桌面上方、两臂之间偏外侧的安全高度。\n"
        "    夹着方块在桌面和盒子之间移动时会经过这里，避免扫到盒子/另一条臂。",
    ),
    "table_approach": (
        "桌面上方接近位",
        "在 table_hover 正上方再高一些（大约一个手掌的高度）。\n"
        "    快速移动会先到这里，再慢速竖直下到 hover，减少冲击和超调。",
    ),
    "table_hover": (
        "桌面观察位",
        "夹爪张开，正对桌上方块上方，腕相机能清楚看到方块。\n"
        "    这一位用来做视觉对中检查；抬起/放下后也回到这里。",
    ),
    "table_pick": (
        "桌面抓取位",
        "从 table_hover 沿 Z 轴竖直向下，让两指正好包住方块两侧、不压桌面。\n"
        "    重要：只动 shoulder_lift / elbow_flex / wrist_flex，\n"
        "    shoulder_pan 和 wrist_roll 要和 hover 基本一致（相差 < 2）。",
    ),
    "box_approach": (
        "盒子上方接近位",
        "在 box_hover 正上方再高一些。",
    ),
    "box_hover": (
        "盒子观察位",
        "夹爪张开，位于盒口正上方，腕相机能看清盒内的方块。",
    ),
    "box_pick": (
        "盒内抓取位",
        "从 box_hover 竖直向下，两指包住盒内方块、不压盒底。\n"
        "    shoulder_pan / wrist_roll 要和 box_hover 基本一致（相差 < 2）。",
    ),
    "box_drop": (
        "盒内释放位",
        "盒内比 box_pick 略高一点点的位置（方块会落下几毫米）。\n"
        "    水平位置要与 box_pick 一致，这样下一轮还能在 box_pick 抓到它。\n"
        "    如果方块放在 box_pick 高度会被压住，就把这里抬高 1~3。",
    ),
}

POSE_ALIAS_TABLE = "table_drop"


def find_video_by_usb(port: str) -> str | None:
    """按 USB 端口路径（例如 "5-2"）找对应的 /dev/videoN。

    /dev/video0..5 的编号在每次插拔后会重新分配，写死路径会在中途
    变成另一台相机（甚至打不开）。USB 端口是物理位置，稳定得多。
    """
    nodes = find_videos_by_usb(port)
    return nodes[0] if nodes else None


def find_videos_by_usb(port: str) -> list[str]:
    """返回该 USB 端口下的全部 /dev/videoN。

    同一个 UVC 相机通常暴露两个节点：一个是采集节点，另一个是元数据
    节点（打不开或读不到帧）。这里全列出来，交给 CameraStream 逐个试。
    """
    base = Path("/sys/class/video4linux")
    if not base.is_dir():
        return []
    out: list[str] = []
    for d in sorted(base.glob("video*")):
        try:
            parts = str((d / "device").resolve()).split("/")
        except Exception:
            continue
        if port in parts:
            out.append(f"/dev/{d.name}")
    return out


def default_calibration_path(arm: str) -> Path:
    """官方 lerobot-calibrate 写出的那条臂的标定文件。

    电机行程一律用官方工具标定，文件落在 lerobot 自己的标定目录里：
      左臂（ACM1，teleop/leader）:
        lerobot-calibrate --teleop.type=so101_leader \\
            --teleop.port=/dev/ttyACM1 --teleop.id=zihao_leader_arm
      右臂（ACM0，follower）:
        lerobot-calibrate --robot.type=so101_follower \\
            --robot.port=/dev/ttyACM0 --robot.id=zihao_follower_arm

    id 换成别的名字时，把 config.json 里 devices.<arm>_calibration_json
    指到对应文件即可。
    """
    base = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
    if arm == "left":
        return base / "teleoperators" / "so_leader" / "zihao_leader_arm.json"
    return base / "robots" / "so_follower" / "zihao_follower_arm.json"


def official_calibrate_cmd(arm: str) -> str:
    """那条臂该用哪条官方标定命令（出错提示里直接抄）。"""
    if arm == "left":
        return ("lerobot-calibrate --teleop.type=so101_leader "
                "--teleop.port=/dev/ttyACM1 --teleop.id=zihao_leader_arm")
    return ("lerobot-calibrate --robot.type=so101_follower "
            "--robot.port=/dev/ttyACM0 --robot.id=zihao_follower_arm")


def calibration_paths(cfg: dict) -> dict[str, Path]:
    """每条臂的电机标定文件路径。

    优先用 devices.<arm>_calibration_json；为了兼容老配置，也认
    devices.calibration_json（一个文件装两条臂，比如 xlerobot.json）。
    """
    dev = cfg.get("devices") or {}
    legacy = dev.get("calibration_json")
    out: dict[str, Path] = {}
    for arm in ARMS:
        p = dev.get(f"{arm}_calibration_json") or legacy
        if p:
            out[arm] = expand_path(p)
    return out


DEFAULTS: dict = {
    "version": 2,
    "devices": {
        # "auto" = 自动识别（按电机数量：8 个电机的是左臂，6 个的是右臂）
        "left_port": "auto",
        "right_port": "auto",
        "left_camera": "/dev/video2",
        "right_camera": "/dev/video0",
        "left_camera_fallbacks": ["/dev/left_wrist_cam", "/dev/video2", "/dev/video3"],
        "right_camera_fallbacks": ["/dev/right_wrist_cam", "/dev/video0", "/dev/video1"],
        # 电机行程只用官方 lerobot-calibrate 标定，这里指向它写出的文件
        "left_calibration_json": str(default_calibration_path("left")),
        "right_calibration_json": str(default_calibration_path("right")),
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 30,
    },
    "detector": {
        "yolo_model": "yoloe-11l-seg.pt",
        "yolo_prompts": ["green cube", "green block", "cube", "green object"],
        "yolo_target_terms": ["cube", "block", "object"],
        "yolo_conf": 0.08,
        "yolo_device": "cuda:0",
        "yolo_imgsz": 640,
        "yolo_interval_s": 0.055,
        "yolo_warmup": True,
        "hsv_low1": [45, 70, 55],
        "hsv_high1": [95, 255, 255],
        "hsv_low2": [45, 70, 55],
        "hsv_high2": [95, 255, 255],
        "min_area": 300,
        "min_area_box": 160,
        "max_area_frac": 0.35,
        "min_color_frac": 0.06,
        "use_yolo": True,
    },
    "motion": {
        "control_hz": 100.0,
        "read_hz": 12.0,
        # 归一化单位/秒。pan/lift/elbow 慢一点，腕部可以快
        "vmax": {
            "shoulder_pan": 55.0,
            "shoulder_lift": 55.0,
            "elbow_flex": 60.0,
            "wrist_flex": 75.0,
            "wrist_roll": 70.0,
        },
        "cruise_scale": 1.0,       # 搬运段整体速度倍率
        "precision_scale": 0.32,   # 最后竖直下探段的速度倍率
        "ease_frac": 0.16,         # 路径两端的缓起缓停比例
        "settle_s": 0.06,          # 轨迹结束后额外等待
        # 实测静差 1.0~1.7 个归一化单位（约 1°），0.9 的容差永远到不了，
        # 每一次移动都会被判成“没到位”。
        "arrive_tol": 3.0,
        "arrive_timeout_s": 0.8,
        "gripper_open_settle_s": 0.28,
        "gripper_close_settle_s": 0.34,
        "release_dwell_s": 0.18,   # 放方块时张爪后原地停留，让方块落稳再抬
        "temp_slow_scale": 0.7,    # 超过 temp_slow_c 后的巡航速度倍率
        "p_gain": 40,              # 关节位置刚度（出厂 20，静差约 1.5°）
        "i_gain": 0,
        "d_gain": 40,
        "gripper_p_gain": 32,
        "gripper_d_gain": 40,
        "gripper_max_torque": 1000,   # 出厂 500，夹方块会滑
        "gripper_torque": 1000,
        "max_overshoot": 5.0,
        "overlap_idle_arm": True,  # 空闲臂趁另一条臂工作时预位
    },
    "grasp": {
        "pos_margin": 0.8,
        "load_min": 25.0,
        "open_hold_margin": 8.0,
        "retries": 3,
        "retry_deepen": 0.8,
        "align_max_iters": 3,
        "align_tol_px": 10.0,
        "align_step_deg": 3.0,
        "align_gain_fallback": [[-4.0, 0.0], [0.0, -22.0]],
        "verify_in_hand_px": 70.0,
        "verify_dst_tol_px": 60.0,
        "require_vision_verify": False,
        "align_roll": False,
        "align_roll_max_deg": 25.0,
        "table_release_lift": 0.0,
        "box_drop_extra_lift": 0.0,
        "drop_tol_px": 25.0,       # 放完方块与 drop_px 的允许像素偏差
        "lift_speed": 0.45,        # 夹住后抬升的速度倍率（快了会甩脱）
        "grip_hold_s": 0.15,       # 合爪后停顿再抬
        "soft_close_step": 8.0,    # 合爪分步，避免一步合死把方块弹开
        "soft_close_settle_s": 0.10,
    },
    "safety": {
        "temp_warn_c": 62.0,
        "temp_slow_c": 68.0,
        "temp_stop_c": 74.0,
        "temp_wait_s": 40.0,
        "stall_restart_s": 180.0,   # 主循环多久没进展就退出让 systemd 重启
        "hard_joint_limit": 100.0,
        "bus_retries": 2,
        "reconnect_after_errors": 6,
        "max_consecutive_failures": 6,
        "cycle_timeout_s": 75.0,
        "phase_timeout_s": 25.0,
        "recover_pause_s": 1.2,
        "startup_temp_wait_s": 30.0,
    },
    "logging": {
        "max_mb": 12,
        "backups": 6,
        "console_level": "INFO",
    },
    "loop": {
        "cycle_pause_s": 0.10,
        # "in_box" = 左臂取出后继续夹着，下一轮直接放入盒子（最快，真正的往复）
        # "table"  = 左臂取出后放回桌面，下一轮重新从桌面抓起（每轮都有视觉确认）
        "mode": "in_box",
        "save_every_n_frames": 0,
        "snapshot_dir": "snapshots",
        "snapshot_every_n_cycles": 10,
        "snapshot_keep": 200,      # 只保留最近这么多张周期快照
    },
    # ------------------------------------------------------------------
    # 视觉伺服（dualarm_cycle.py / servo.py）
    # ------------------------------------------------------------------
    "servo": {
        "enable": True,
        "state_file": "servo_state.json",
        "probe_deg": 6.0,             # 测雅可比的微探针角度（太小会被检测噪声淹没）
        "target_tol_px": 7.0,         # hover 对准容差
        "pick_gate_px": 30.0,         # 合爪前允许的底部像素偏差
        "pick_gate_fresh_s": 0.45,    # 合爪前检测的新鲜度上限
        "max_iters": 6,               # hover 对准最多迭代次数
        "max_step_deg": 7.0,          # 单步修正上限
        "max_total_deg": 16.0,        # 累计修正上限（别离标定位姿太远）
        "hover_abort_px": 40.0,       # hover 残差过大就换另一条臂，不下探推方块
        "damping": 0.15,              # 阻尼最小二乘的 λ
        "descent_steps": 4,           # 下探分几段
        "descent_tol_px": 12.0,       # 下探跟踪的像素容差
        "descent_speed": 0.5,         # 下探速度倍率
        "learn_ema": 0.25,            # 学习值 EMA
        "learn_max_jump_px": 45.0,    # 单次学习允许的最大跳变
        "reject_jump_px": 90.0,       # 相对预测目标的跳变拒绝阈值
        "reject_area_ratio": 2.5,     # 面积突变拒绝比例
        "require_visual_before_close": True,
        "stable_px": 18.0,            # 连续两帧位置差超过它就算不可靠
        "box_gate_enable": True,      # 盒内检测的空间合理性过滤
        "box_gate_px": 200.0,         # 离盒子中心超过这么多像素就不是方块
        "bootstrap_center_px": [340.0, 330.0],
        "bootstrap_radius_px": 220.0,
        "jacobian_min_pan_gain": 1.0, # pan 的像素增益不能太小（否则是误检）
        "jacobian_min_det": 0.5,      # 雅可比行列式下限
        "pick_retries": 5,            # 一次抓取里最多重试几轮
        "gate_reset_after": 3,        # 闸门连续失败几次后清学习值重新引导
    },
    # ------------------------------------------------------------------
    # 流畅度（dualarm_cycle.py 在启动时覆盖到 motion/grasp）
    # ------------------------------------------------------------------
    "smooth": {
        "apply": True,
        "settle_s": 0.02,
        "arrive_tol": 4.0,
        "arrive_timeout_s": 0.25,
        "descent_tol": 6.0,
        "descent_speed": 0.5,
        "lift_speed": 0.5,
        "cycle_pause_s": 0.0,
        "release_dwell_s": 0.20,
        "gripper_open_settle_s": 0.20,
        "gripper_close_settle_s": 0.28,
        "grip_hold_s": 0.25,
        "load_settle_s": 0.20,
        "place_verify": True,
        "place_retries": 1,
        "place_tol_px": 55.0,
        "preposition": True,
    },
    "arms": {
        "left": {
            "poses": {},
            "gripper_open": None,
            "gripper_closed": None,
            "table_pick_height_offset": 0.0,
            "box_pick_height_offset": 0.0,
            "gains": {
                "hover_px_per_deg": None,
                "roll_deg_per_deg": None,
                "align_px": None,
                "held_px": None,
                "box_roi": None,
                "drop_px": None,
                "drop_offset": [0.0, 0.0],
            },
        },
        "right": {
            "poses": {},
            "gripper_open": None,
            "gripper_closed": None,
            "table_pick_height_offset": 0.0,
            "box_pick_height_offset": 0.0,
            "gains": {
                "hover_px_per_deg": None,
                "roll_deg_per_deg": None,
                "align_px": None,
                "held_px": None,
                "box_roi": None,
                "drop_px": None,
                "drop_offset": [0.0, 0.0],
            },
        },
    },
}


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path.exists():
        try:
            raw = json.loads(path.read_text())
        except Exception as exc:
            raise SystemExit(f"config.json 解析失败: {exc}")
        cfg = deep_merge(cfg, raw)
    return cfg


def save_config(patch: dict, path: Path = CONFIG_PATH) -> None:
    """把 patch 深度合并进现有 config.json（原子写）。"""
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text())
        except Exception:
            current = {}
    merged = deep_merge(current, patch)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def expand_path(p) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


# ---------------------------------------------------------------------------
# 位姿
# ---------------------------------------------------------------------------

def arm_cfg(cfg: dict, arm: str) -> dict:
    return cfg["arms"][arm]


def get_pose(cfg: dict, arm: str, key: str, required: bool = True) -> Optional[np.ndarray]:
    poses = arm_cfg(cfg, arm).get("poses") or {}
    value = poses.get(key)
    if value is None:
        if required:
            raise SystemExit(
                f"[{arm}] 缺少位姿 '{key}'，请先运行: python calibrate2arm.py --arm {arm}")
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape != (NJ,):
        raise SystemExit(f"[{arm}.{key}] 需要 {NJ} 个数，实际 {arr.shape}")
    return arr


def set_pose(cfg: dict, arm: str, key: str, q: Sequence[float]) -> None:
    arr = np.asarray(q, dtype=float)
    if arr.shape != (NJ,):
        raise ValueError(f"位姿 {key} 形状错误: {arr.shape}")
    cfg["arms"][arm].setdefault("poses", {})[key] = [round(float(v), 4) for v in arr]


def offset_pose(q: Sequence[float], **deltas) -> np.ndarray:
    """按关节名偏移位姿，例如 offset_pose(q, shoulder_lift=-4)。"""
    out = np.asarray(q, dtype=float).copy()
    for name, delta in deltas.items():
        if name not in JOINTS:
            raise KeyError(f"未知关节 {name}")
        out[JOINTS.index(name)] += float(delta)
    return out


def missing_poses(cfg: dict, arm: str) -> list[str]:
    poses = arm_cfg(cfg, arm).get("poses") or {}
    return [k for k in POSE_KEYS if poses.get(k) is None]


def saturated_joints(q: Sequence[float], tol: float = 1e-3) -> list[str]:
    """返回读数被截断的关节名。

    lerobot 的 _normalize 把超出标定范围的读数钳到 ±100。被钳过之后，
    不同的物理高度会变成同一个数，下探就失去意义（等于按同一点位反复
    往限位里挤），所以这种点必须重标。

    判据只能是“正好等于 ±100”：99.9 是离边界 0.1 个归一化单位（约 0.2°）
    的有效读数，高度仍然分得开，不能当成截断拒收。容差放在 1e-3，是为了
    容忍 JSON 里 round(…, 4) 带来的误差。
    """
    out = []
    for i, name in enumerate(JOINTS):
        if abs(float(q[i])) >= 100.0 - tol:
            out.append(f"{name}({'+100' if q[i] > 0 else '-100'})")
    return out


def pose_saturation_report(cfg: dict, arm: str) -> list[str]:
    """列出所有饱和的位姿，供启动前检查。"""
    msgs = []
    for key in POSE_KEYS:
        q = get_pose(cfg, arm, key, required=False)
        if q is None:
            continue
        bad = saturated_joints(q)
        if bad:
            msgs.append(f"  [{arm}] {key}: {', '.join(bad)}")
    return msgs


def duplicate_pose_report(cfg: dict, arm: str, tol: float = 3.0) -> list[str]:
    """检查 approach 位和 hover 位是否几乎重合。

    approach 的意义就是“比 hover 高一个手掌”，先到这里再竖直慢速下探。
    两者重合时这段减速完全消失，会带着巡航速度直接砸到方块/盒子；
    而且通常说明其中一次是按错回车录的。
    """
    msgs = []
    for approach, hover in (("table_approach", "table_hover"),
                            ("box_approach", "box_hover")):
        a = get_pose(cfg, arm, approach, required=False)
        h = get_pose(cfg, arm, hover, required=False)
        if a is None or h is None:
            continue
        d = float(np.abs(a - h).max())
        if d < tol:
            msgs.append(f"  [{arm}] {approach} 与 {hover} 只差 {d:.1f}，"
                        f"基本是同一个姿态；建议重标 {approach}（要比 {hover} 高一截）")
    return msgs


def tight_pose_report(cfg: dict, arm: str, fw: dict,
                      margin: float = 5.0) -> list[str]:
    """列出离标定边界过近的位姿（归一化单位）。

    |q| 还没到 100，所以不算截断，也不该拒绝启动；但只剩几个单位时，
    轨迹插值和视觉微调很容易越过边界，被 np.clip 压平——表现为那一段
    “发指令但不动”。属于该提醒、不该硬拦的情况。
    """
    prefix = ARM_PREFIX[arm]
    msgs = []
    for key in POSE_KEYS:
        q = get_pose(cfg, arm, key, required=False)
        if q is None:
            continue
        tight = []
        for i, joint in enumerate(JOINTS):
            c = fw.get(f"{prefix}_{joint}")
            if c is None:
                continue
            span = max(1.0, float(c.range_max - c.range_min))
            raw = (float(q[i]) + 100.0) / 200.0 * span + c.range_min
            m = min((raw - c.range_min) / span * 200.0,
                    (c.range_max - raw) / span * 200.0)
            if m < margin:
                tight.append(f"{joint}({m:.1f})")
        if tight:
            msgs.append(f"  [{arm}] {key}: {', '.join(tight)}")
    return msgs


def calib_fingerprint(fw: dict) -> str:
    """电机标定的指纹：只取 id/范围/homing，用来判断点位属于哪套标定。

    同一条臂换了电机标定（重标、换臂、换机器）后，归一化坐标系整体
    平移/缩放，旧点位就全失效了。给每个点位都记一个指纹，启动时逐个
    比对，比事后发现臂撞限位便宜得多。
    """
    items = []
    for name in sorted(fw):
        c = fw[name]
        items.append(f"{name}:{getattr(c, 'id', '')}:{getattr(c, 'range_min', '')}:"
                     f"{getattr(c, 'range_max', '')}:{getattr(c, 'homing_offset', '')}")
    return "|".join(items)


def calib_fingerprint_short(fw: dict) -> str:
    """指纹的短哈希，存进 config 用（完整指纹太长）。"""
    import hashlib
    return hashlib.sha1(calib_fingerprint(fw).encode()).hexdigest()[:16]


def get_pose_calib(cfg: dict, arm: str) -> dict:
    return (cfg["arms"][arm].get("pose_calib") or {})


def set_pose_calib(cfg: dict, arm: str, key: str, fw: dict) -> None:
    """记下这个点位是在哪套电机标定下录的。"""
    cfg["arms"][arm].setdefault("pose_calib", {})[key] = calib_fingerprint_short(fw)


def pose_calib_mismatch(cfg: dict, arm: str, fw: dict) -> tuple[list[str], list[str]]:
    """逐点位比对“录制时的电机标定”和现在是否一致。

    返回 (与当前标定不同, 没有留过指纹)。逐点比对而不是整条臂一个指纹，
    因为标定可能中途中断，导致一条臂上新旧点位混在一起——整臂一个指纹
    查不出这种情况。
    """
    now = calib_fingerprint_short(fw)
    recorded = get_pose_calib(cfg, arm)
    differs: list[str] = []
    unknown: list[str] = []
    for key in POSE_KEYS:
        if get_pose(cfg, arm, key, required=False) is None:
            continue
        got = recorded.get(key)
        if got is None:
            unknown.append(key)
        elif got != now:
            differs.append(key)
    return differs, unknown


def pose_delta_report(cfg: dict, arm: str) -> list[str]:
    """检查 hover -> pick 是否接近纯竖直（pan/roll 不应变化）。

    符号约定（与 lerobot 归一化一致）：shoulder_lift 增大 = 手臂降低。
    所以从 hover 下探到 pick 时，lift 通常是正值。
    """
    msgs = []
    for hover_key, pick_key in (("table_hover", "table_pick"), ("box_hover", "box_pick")):
        h = get_pose(cfg, arm, hover_key, required=False)
        p = get_pose(cfg, arm, pick_key, required=False)
        if h is None or p is None:
            continue
        d = p - h
        msgs.append(
            f"  {hover_key}->{pick_key}: pan={d[0]:+.1f} lift={d[1]:+.1f} "
            f"elbow={d[2]:+.1f} wrist={d[3]:+.1f} roll={d[4]:+.1f}"
            f"  (lift>0 表示下探/降低)")
        for idx, name in ((0, "shoulder_pan"), (4, "wrist_roll")):
            if abs(d[idx]) > 2.0:
                msgs.append(f"    ! {name} 变化 {d[idx]:+.1f}，这段不是竖直下降，"
                            f"抓取会偏；建议重标 {pick_key}")
        if abs(d[1]) < 1.0:
            msgs.append(f"    ! shoulder_lift 只变化 {d[1]:+.1f}，"
                        f"{pick_key} 可能和 {hover_key} 几乎同高，下探无效")
    return msgs


# ---------------------------------------------------------------------------
# 数值/轨迹
# ---------------------------------------------------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def fold_angle(angle: float, w: float, h: float) -> float:
    """把 minAreaRect 的角度折叠到 [-45, 45)；正方形有 90° 对称。"""
    if w < h:
        angle += 90.0
    return ((angle + 45.0) % 90.0) - 45.0


def in_roi(pt, roi) -> bool:
    if roi is None:
        return True
    x, y = float(pt[0]), float(pt[1])
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def smootherstep(x: float) -> float:
    x = clamp(float(x), 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _catmull_rom(points: np.ndarray, samples_per_seg: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 1:
        return pts.copy()
    if len(pts) == 2:
        # 单段也必须采样成密集点。只给两个点时，time_parameterize 的
        # 端点缓动作用在整段上，整条动作会被压到巡航速度的 1/50。
        ts = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
        line = pts[0][None, :] + (pts[1] - pts[0])[None, :] * ts
        return np.vstack([line, pts[1][None, :]])
    padded = np.vstack([pts[0], pts, pts[-1]])
    chunks = []
    for i in range(len(padded) - 3):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        ts = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
        t2 = ts * ts
        t3 = t2 * ts
        seg = 0.5 * ((2.0 * p1) + (-p0 + p2) * ts
                     + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                     + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
        chunks.append(seg)
    chunks.append(pts[-1:])
    return np.vstack(chunks)


def build_path(waypoints: Sequence[Sequence[float]], max_overshoot: float = 5.0,
               samples_per_seg: int = 40) -> np.ndarray:
    """生成平滑路径；如果 Catmull-Rom 冲得太远就退回线性插值。"""
    pts = np.asarray(waypoints, dtype=float)
    if len(pts) == 1:
        return pts.copy()
    path = _catmull_rom(pts, samples_per_seg)
    # 逐段检查：偏离该段两端点范围太多 -> 该段用线性
    ok = True
    seg_n = len(path) // (len(pts) - 1) if len(pts) > 1 else len(path)
    for i in range(len(pts) - 1):
        lo = np.minimum(pts[i], pts[i + 1]) - max_overshoot
        hi = np.maximum(pts[i], pts[i + 1]) + max_overshoot
        a = i * seg_n
        b = len(path) if i == len(pts) - 2 else (i + 1) * seg_n + 1
        chunk = path[a:b]
        if np.any(chunk < lo - 1e-9) or np.any(chunk > hi + 1e-9):
            ok = False
            break
    if not ok:
        chunks = []
        for i in range(len(pts) - 1):
            ts = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
            chunks.append(pts[i][None, :] + (pts[i + 1] - pts[i])[None, :] * ts)
        chunks.append(pts[-1:])
        path = np.vstack(chunks)
    return np.clip(path, -100.0, 100.0)


def time_parameterize(path: np.ndarray, vmax: np.ndarray,
                      ease_frac: float = 0.16) -> tuple[np.ndarray, np.ndarray]:
    """把几何路径变成 (时间表, 路径)。速度在中间恒定、两端平滑加减速。"""
    path = np.asarray(path, dtype=float)
    vmax = np.maximum(np.asarray(vmax, dtype=float), 1e-3)
    if len(path) < 2:
        return np.zeros(1), path
    steps = np.max(np.abs(np.diff(path, axis=0)) / vmax[None, :], axis=1)
    natural = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(natural[-1])
    if total < 1e-9:
        return np.zeros(len(path)), path
    u = natural / total
    a = max(1e-3, float(ease_frac))
    ramp = np.minimum(1.0, np.minimum(u / a, (1.0 - u) / a))
    w = np.array([smootherstep(v) for v in ramp])
    mid = 0.5 * (w[:-1] + w[1:])
    # 速度系数下限。smootherstep 在 u=0 处趋近 0，若按 0 处理，端点那几段
    # 的时长会发散（旧值 2e-2 会让整段慢 50 倍）。0.30 = 端点是巡航速度的
    # 30%，剩下的加速度突变由电机自身的 Acceleration 限制吃掉。
    mid = np.maximum(mid, 0.30)
    dt = steps / mid
    times = np.concatenate([[0.0], np.cumsum(dt)])
    return times, path


def sample_path(times: np.ndarray, path: np.ndarray, t: float) -> np.ndarray:
    if len(path) == 1:
        return path[0].copy()
    if t <= times[0]:
        return path[0].copy()
    if t >= times[-1]:
        return path[-1].copy()
    return np.array([np.interp(t, times, path[:, j]) for j in range(path.shape[1])])


def vmax_vector(cfg: dict) -> np.ndarray:
    vmax = cfg["motion"]["vmax"]
    return np.array([float(vmax[j]) for j in JOINTS], dtype=float)


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

def setup_logging(cfg: dict, name: str = "s2b2") -> logging.Logger:
    from logging.handlers import RotatingFileHandler

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")

    fh = RotatingFileHandler(
        LOG_DIR / f"{name}.log",
        maxBytes=int(cfg["logging"]["max_mb"]) * 1024 * 1024,
        backupCount=int(cfg["logging"]["backups"]),
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    level = getattr(logging, str(cfg["logging"].get("console_level", "INFO")).upper(), logging.INFO)
    sh.setLevel(level)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def fmt_q(q: Iterable[float], nd: int = 1) -> str:
    return "[" + " ".join(f"{v:+.{nd}f}" for v in q) + "]"


class StopRequested(RuntimeError):
    """收到 SIGINT/SIGTERM。"""


class PhaseError(RuntimeError):
    """某个动作阶段失败，可以重试或进入恢复。"""


class BusError(RuntimeError):
    """串口通信失败。"""


class TargetMissing(PhaseError):
    """视觉上找不到目标。"""


class GraspFailed(PhaseError):
    """夹爪没夹住。"""


def now() -> float:
    return time.perf_counter()
