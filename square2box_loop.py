#!/usr/bin/env python
"""square2box_loop.py — 24/7 无人值守抓取循环（YOLO + 视觉伺服）。

与 square2box.py 的区别：
  - 检测：YOLOE 开放词表分割（本地模型）+ HSV 精修/兜底，不再只靠 HSV。
  - 标定：沿用 config.json 的 hover/grasp/box 位姿；新增 box_pick/table_drop 可选位姿。
  - 运动：50Hz 平滑插值（smootherstep），不再每步读总线；到位用反馈确认。
  - 循环：桌面抓取 → 放入盒子 → 从盒子抓出 → 放回桌面 → 重复。
  - 可靠性：相机看门狗、机器人重连、温度保护、异常恢复、轮转日志。
  - 平滑：视觉伺服按相机帧率跑，减少固定 sleep；抓取验证用位置 + 负载 + 视觉。

用法（conda env: hupy3.12）:
  python square2box_loop.py self-test           # 只检查，不动电机
  python square2box_loop.py preview --seconds 10
  python square2box_loop.py calibrate-extra     # 一次性录 box_pick/table_drop/box_look/home
  python square2box_loop.py calibrate-depth     # 一次性探测深度雅可比（会动）
  python square2box_loop.py run --cycles 0      # 无限循环，Ctrl+C 停止
  python square2box_loop.py run --dry-run       # 只开相机/YOLO，不动电机

安全设计：
  - 所有运动都有速度上限、关节限幅、超时和停止事件。
  - 异常/退出时先开爪、再下电力矩。
  - 温度过高自动卸力等待；相机/总线失联自动重连。
"""

from __future__ import annotations

import argparse
import builtins
import csv
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
LOG_DIR = HERE / "logs"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINT_KEYS = [f"left_arm_{j}.pos" for j in JOINTS]
GRIPPER_KEY = "left_arm_gripper.pos"
CAM_KEY = "left_wrist"
CAM_DEVICE = "/dev/left_wrist_cam"

try:
    from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
except ImportError:
    sys.path.insert(0, "/home/zhoulin/hu/lerobot/src")
    from lerobot.robots.xlerobot import XLerobot, XLerobotConfig


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

LOOP_DEFAULTS = {
    # 设备（udev 符号链接缺失时自动回退到原始设备）
    "camera_device": "/dev/left_wrist_cam",
    "left_arm_port": "/dev/left_arm",
    "right_arm_port": "/dev/right_arm",
    # 颜色兜底（正式配置在 config.json 里覆盖）
    "hsv_low1": [50, 80, 70],
    "hsv_high1": [85, 255, 255],
    "hsv_low2": [50, 80, 70],
    "hsv_high2": [85, 255, 255],
    # YOLO
    "yolo_model": "yoloe-11l-seg.pt",
    "yolo_prompts": ["green cube", "green block", "cube"],
    "yolo_target_terms": ["cube", "block"],
    "yolo_conf": 0.06,
    "yolo_device": 0,
    "yolo_imgsz": 640,
    "yolo_interval_s": 0.08,
    # 检测过滤
    "min_area": 350,
    "min_area_box": 200,
    "max_area_frac": 0.35,
    "min_color_frac": 0.08,
    "box_target_pixel": [320, 240],
    "box_roi_radius": 150,
    # 新增标定（None = 由现有位姿推导）
    "box_pick_joints": None,
    "table_drop_joints": None,
    "box_look_joints": None,
    "home_joints": None,
    "safe_joints": None,
    "measure_pan_offset": 20.0,
    "depth_gain_matrix": None,
    "roll_slope_deg_per_deg": None,
    "box_pick_deepen_deg": 2.0,
    "box_pick_height_offset_deg": 0.0,
    "box_look_pixel": None,
    "box_pick_pixel": None,
    "table_drop_lift_deg": -1.0,
    "grasp_height_offset_deg": 6.0,
    # 运动
    "max_vel_deg_s": 45.0,
    "control_hz": 50.0,
    "servo_hz": 25.0,
    "servo_kp": 0.7,
    "servo_tol_px": 6.0,
    "servo_step_deg": 2.0,
    "servo_max_iters": 45,
    "depth_tol_px": 12.0,
    "depth_step_deg": 3.0,
    "depth_max_iters": 10,
    "depth_gain": 0.8,
    "grasp_retries": 3,
    "place_retries": 2,
    "gripper_settle_s": 0.70,
    "grasp_pos_margin": 0.6,
    "grasp_load_min": 20.0,
    "cycle_pause_s": 0.25,
    "max_cycle_time_s": 90.0,
    "max_consecutive_failures": 5,
    "recover_wait_s": 3.0,
    "temp_limit_c": 58.0,
    "temp_wait_s": 45.0,
    "temp_check_s": 5.0,
    "log_max_mb": 10,
    "log_backups": 5,
}


def load_config() -> dict:
    cfg = dict(LOOP_DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    return cfg


def save_config(cfg: dict) -> None:
    current = {}
    if CONFIG_PATH.exists():
        try:
            current = json.loads(CONFIG_PATH.read_text())
        except Exception:
            current = {}
    current.update(cfg)
    CONFIG_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False))
    logging.getLogger("s2b").info("配置已保存: %s", CONFIG_PATH)


def require_pose(cfg: dict, key: str) -> np.ndarray:
    v = cfg.get(key)
    if v is None:
        raise SystemExit(f"缺少位姿 {key}，请先运行 calibrate 或 calibrate-extra")
    arr = np.asarray(v, dtype=float)
    if arr.shape != (len(JOINTS),):
        raise SystemExit(f"{key} 必须是 {len(JOINTS)} 个数，实际 {arr.shape}")
    return arr


def resolve_device(preferred, fallbacks) -> str:
    p = Path(str(preferred))
    if p.exists():
        return str(p)
    for f in fallbacks:
        if Path(f).exists():
            return str(f)
    return str(p)


def camera_device(cfg: dict) -> str:
    return resolve_device(cfg.get("camera_device", CAM_DEVICE),
                          ["/dev/video2", "/dev/video0"])


def arm_ports(cfg: dict):
    # 当前接线：左臂（8 电机，含头）在 ttyACM1，右臂（6 电机）在 ttyACM0
    left = resolve_device(cfg.get("left_arm_port", "/dev/left_arm"),
                          ["/dev/ttyACM1", "/dev/ttyACM0"])
    right = resolve_device(cfg.get("right_arm_port", "/dev/right_arm"),
                           ["/dev/ttyACM0", "/dev/ttyACM1"])
    return left, right


# ---------------------------------------------------------------------------
# 数学/工具
# ---------------------------------------------------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smootherstep(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def fold_angle(angle: float, w: float, h: float) -> float:
    """把 minAreaRect 的角度折叠到 [-45, 45)，正方形对称。"""
    if w < h:
        angle += 90.0
    return ((angle + 45.0) % 90.0) - 45.0


def bbox_center(bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


def in_roi(pt, roi) -> bool:
    if roi is None:
        return True
    x, y = float(pt[0]), float(pt[1])
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


class CameraError(RuntimeError):
    pass


class TargetLost(RuntimeError):
    pass


class ServoTimeout(RuntimeError):
    pass


class CycleError(RuntimeError):
    pass


class StopRequested(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 相机线程
# ---------------------------------------------------------------------------

class CameraStream:
    def __init__(self, device=CAM_DEVICE, width=640, height=480, fps=30,
                 logger: Optional[logging.Logger] = None):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.log = logger or logging.getLogger("s2b")
        self._frame = None
        self._ts = 0.0
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._reopen = False
        self._cap = None
        self._thread = None
        self._fps_est = 0.0
        self._frame_times = []

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
        self._thread.start()
        self.log.info("相机线程启动: %s", self.device)

    def _open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        cap = None
        while not self._stop.is_set():
            if self._reopen:
                self._reopen = False
                if cap is not None:
                    cap.release()
                cap = None
                self.log.warning("相机重连中...")
            if cap is None or not cap.isOpened():
                cap = self._open()
                self._cap = cap
                if not cap.isOpened():
                    self.log.error("打不开相机 %s，0.5s 后重试", self.device)
                    time.sleep(0.5)
                    continue
                self.log.info("相机已打开: %dx%d@%d", self.width, self.height, self.fps)
            ok, frame = cap.read()
            if not ok or frame is None:
                self.log.warning("相机读帧失败，重连")
                cap.release()
                cap = None
                self._cap = None
                time.sleep(0.2)
                continue
            now = time.time()
            with self._cond:
                self._frame = frame
                self._ts = now
                self._seq += 1
                self._frame_times.append(now)
                if len(self._frame_times) > 30:
                    self._frame_times = self._frame_times[-30:]
                if len(self._frame_times) >= 2:
                    dt = self._frame_times[-1] - self._frame_times[0]
                    self._fps_est = (len(self._frame_times) - 1) / dt if dt > 0 else 0.0
                self._cond.notify_all()
        if cap is not None:
            cap.release()
            self._cap = None

    def read(self, timeout=2.0, max_age=1.0) -> np.ndarray:
        with self._cond:
            seq = self._seq
            self._cond.wait_for(lambda: self._seq > seq or self._stop.is_set(), timeout)
            if self._frame is None:
                raise CameraError("相机还没有画面")
            if time.time() - self._ts > max_age:
                raise CameraError(f"相机画面过期 {time.time() - self._ts:.1f}s")
            return self._frame.copy()

    def latest(self):
        with self._cond:
            return (None if self._frame is None else self._frame.copy(), self._ts)

    def age(self) -> float:
        with self._cond:
            return float("inf") if self._ts == 0 else time.time() - self._ts

    @property
    def fps_est(self) -> float:
        with self._cond:
            return self._fps_est

    def reconnect(self):
        self._reopen = True
        cap = self._cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.log.info("相机线程停止")


# ---------------------------------------------------------------------------
# 检测
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    cx: float
    cy: float
    area: float
    angle: float
    bbox: tuple
    conf: float
    source: str
    ts: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=float)


class Detector:
    """YOLOE 开放词表分割 + HSV 精修/兜底。"""

    def __init__(self, cfg: dict, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.log = logger or logging.getLogger("s2b")
        self.model = None
        self.names = {}
        self.is_yoloe = False
        self.last_yolo_ts = 0.0
        self.cached: Optional[Detection] = None
        self._depth_warned = False

        self.hsv_low1 = np.array(cfg["hsv_low1"], dtype=np.uint8)
        self.hsv_high1 = np.array(cfg["hsv_high1"], dtype=np.uint8)
        self.hsv_low2 = np.array(cfg["hsv_low2"], dtype=np.uint8)
        self.hsv_high2 = np.array(cfg["hsv_high2"], dtype=np.uint8)
        self.min_area = int(cfg["min_area"])
        self.min_area_box = int(cfg["min_area_box"])
        self.max_area_frac = float(cfg["max_area_frac"])
        self.min_color_frac = float(cfg["min_color_frac"])
        self.target_terms = [str(x).lower() for x in cfg["yolo_target_terms"]]
        self._load_model()

    # ---- 模型 ----
    def _load_model(self):
        path = str(self.cfg.get("yolo_model") or "")
        candidates = [Path(path), HERE / path, HERE / "weights" / "cube_yolo.pt",
                      HERE / "yoloe-11l-seg.pt"]
        model_path = next((p for p in candidates if p and p.exists()), None)
        if model_path is None:
            self.log.warning("找不到 YOLO 模型 %s，使用 HSV 兜底", path)
            return
        try:
            from ultralytics import YOLOE, YOLO
            if "yoloe" in model_path.name.lower():
                self._ensure_text_encoder()
                self.model = YOLOE(str(model_path))
                prompts = list(self.cfg["yolo_prompts"])
                self.model.set_classes(prompts, self.model.get_text_pe(prompts))
                self.is_yoloe = True
            else:
                self.model = YOLO(str(model_path))
                self.is_yoloe = False
            self.names = self.model.names
            self.log.info("YOLO 模型已加载: %s (%s)", model_path.name,
                          "YOLOE" if self.is_yoloe else "YOLO")
        except Exception as e:
            self.model = None
            self.log.exception("YOLO 加载失败，使用 HSV 兜底: %s", e)

    def _ensure_text_encoder(self):
        """YOLOE 需要 mobileclip_blt.ts；优先在工作目录建软链接。"""
        target = Path("mobileclip_blt.ts")
        if target.exists():
            return
        candidates = [
            HERE / "mobileclip_blt.ts",
            Path("/home/zhoulin/hu/lerobot/examples/mobileclip_blt.ts"),
            Path("/home/zhoulin/hu/lerobot/mobileclip_blt.ts"),
        ]
        for c in candidates:
            if c.exists() and c.resolve() != target.resolve():
                try:
                    target.symlink_to(c)
                    self.log.info("已链接 YOLOE 文本编码器: %s -> %s", target, c)
                    return
                except FileExistsError:
                    return
                except Exception as e:
                    self.log.warning("链接文本编码器失败: %s", e)
        self.log.warning("找不到 mobileclip_blt.ts，YOLOE 可能加载失败")

    # ---- 对外 ----
    def detect(self, frame: np.ndarray, roi=None) -> Optional[Detection]:
        now = time.time()
        if self.model is not None and (
            self.cached is None or now - self.last_yolo_ts >= float(self.cfg["yolo_interval_s"])
        ):
            self.cached = self._predict(frame, roi)
            self.last_yolo_ts = now
        if self.cached is not None:
            refined = self._refine_hsv(frame, self.cached, roi)
            if refined is not None:
                return refined
        return self._detect_hsv(frame, roi)

    def annotate(self, frame: np.ndarray, det: Optional[Detection]):
        if det is None:
            cv2.putText(frame, "NO TARGET", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), 2)
            return frame
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.drawMarker(frame, (int(det.cx), int(det.cy)), (0, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
        txt = f"{det.source} {det.conf:.2f} ang={det.angle:.1f} area={det.area:.0f}"
        cv2.putText(frame, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2)
        return frame

    # ---- YOLO ----
    def _predict(self, frame: np.ndarray, roi=None) -> Optional[Detection]:
        if self.model is None:
            return None
        try:
            r = self.model.predict(
                frame,
                conf=float(self.cfg["yolo_conf"]),
                device=self.cfg["yolo_device"],
                imgsz=int(self.cfg["yolo_imgsz"]),
                verbose=False,
            )[0]
        except Exception as e:
            self.log.warning("YOLO 推理失败: %s", e)
            return None
        if r.boxes is None or len(r.boxes) == 0:
            return None
        h, w = frame.shape[:2]
        best = None
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls)
            name = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else (
                self.names[cls_id] if isinstance(self.names, (list, tuple)) and 0 <= cls_id < len(self.names)
                else str(cls_id))
            if not self._is_target(str(name)):
                continue
            conf = float(box.conf)
            if conf < float(self.cfg["yolo_conf"]):
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            if roi is not None and not in_roi(((x1 + x2) / 2, (y1 + y2) / 2), roi):
                continue
            if r.masks is not None and i < len(r.masks.xy):
                poly = np.asarray(r.masks.xy[i], dtype=np.float32).reshape(-1, 1, 2)
                area = float(cv2.contourArea(poly))
                if area < self.min_area * 0.5:
                    continue
                m = cv2.moments(poly)
                if m["m00"] <= 1e-6:
                    continue
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                (_, _), (rw, rh), ang = cv2.minAreaRect(poly)
                angle = fold_angle(ang, rw, rh)
            else:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                angle = 0.0
            if area > h * w * self.max_area_frac:
                continue
            det = Detection(cx, cy, area, angle, (x1, y1, x2, y2),
                            conf, "yolo", time.time())
            if best is None or det.conf > best.conf:
                best = det
        return best

    def _is_target(self, name: str) -> bool:
        s = name.lower()
        if "box" in s:
            return False
        return any(t in s for t in self.target_terms)

    # ---- HSV ----
    def _hsv_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.hsv_low1, self.hsv_high1)
        m2 = cv2.inRange(hsv, self.hsv_low2, self.hsv_high2)
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return mask

    def _refine_hsv(self, frame: np.ndarray, det: Detection, roi=None) -> Optional[Detection]:
        x1, y1, x2, y2 = det.bbox
        pad = 12
        h, w = frame.shape[:2]
        x1 = int(clamp(x1 - pad, 0, w - 1))
        y1 = int(clamp(y1 - pad, 0, h - 1))
        x2 = int(clamp(x2 + pad, 0, w))
        y2 = int(clamp(y2 + pad, 0, h))
        mask = self._hsv_mask(frame)
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255
        if roi is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in roi]
            roi_mask[:max(0, ry1), :] = 0
            roi_mask[ry2:, :] = 0
            roi_mask[:, :max(0, rx1)] = 0
            roi_mask[:, rx2:] = 0
        mask = cv2.bitwise_and(mask, roi_mask)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            # YOLO 框里没有目标颜色 -> 可能是误检
            frac = self._color_frac(frame, det.bbox)
            if frac < self.min_color_frac:
                return None
            return det
        c = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        if area < max(self.min_area * 0.4, 100):
            return det
        m = cv2.moments(c)
        if m["m00"] <= 1e-6:
            return det
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        (_, _), (rw, rh), ang = cv2.minAreaRect(c)
        angle = fold_angle(ang, rw, rh)
        bx, by, bw, bh = cv2.boundingRect(c)
        return Detection(cx, cy, area, angle, (bx, by, bx + bw, by + bh),
                         det.conf, "yolo+hsv", time.time())

    def _color_frac(self, frame: np.ndarray, bbox) -> float:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        mask = self._hsv_mask(frame[y1:y2, x1:x2])
        return float(np.count_nonzero(mask)) / float((x2 - x1) * (y2 - y1))

    def _detect_hsv(self, frame: np.ndarray, roi=None) -> Optional[Detection]:
        mask = self._hsv_mask(frame)
        if roi is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in roi]
            h, w = mask.shape[:2]
            keep = np.zeros_like(mask)
            keep[max(0, ry1):min(h, ry2), max(0, rx1):min(w, rx2)] = 255
            mask = cv2.bitwise_and(mask, keep)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        min_area = self.min_area_box if roi is not None else self.min_area
        if area < min_area:
            return None
        m = cv2.moments(c)
        if m["m00"] <= 1e-6:
            return None
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        (_, _), (rw, rh), ang = cv2.minAreaRect(c)
        angle = fold_angle(ang, rw, rh)
        bx, by, bw, bh = cv2.boundingRect(c)
        return Detection(cx, cy, area, angle, (bx, by, bx + bw, by + bh),
                         1.0, "hsv", time.time())


# ---------------------------------------------------------------------------
# 机械臂
# ---------------------------------------------------------------------------

class Robot:
    def __init__(self, cfg: dict, stop_event: threading.Event,
                 logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.stop_event = stop_event
        self.log = logger or logging.getLogger("s2b")
        self.robot: Optional[XLerobot] = None
        self.gripper = float(cfg["gripper_open"])
        self._bus_lock = threading.RLock()

    def connect(self):
        with self._bus_lock:
            if self.robot is not None:
                self.disconnect()
            cfg = XLerobotConfig(use_head=False, use_base=False)
            left_port, right_port = arm_ports(self.cfg)
            cfg.port1 = left_port
            cfg.port2 = right_port
            cfg.cameras = {}
            self.log.info("串口: left=%s right=%s", left_port, right_port)
            self.robot = XLerobot(cfg)
            orig = builtins.input
            builtins.input = lambda *a, **k: ""
            try:
                self.robot.connect()
            finally:
                builtins.input = orig
            try:
                self.robot.bus1.enable_torque(self.robot.left_arm_motors)
            except Exception as e:
                self.log.warning("整体使能力矩失败: %s，尝试逐关节", e)
                for m in self.robot.left_arm_motors:
                    try:
                        self.robot.bus1.enable_torque(m)
                    except Exception as e2:
                        self.log.error("使能 %s 失败: %s", m, e2)
            self.log.info("机械臂已连接")

    def disconnect(self):
        with self._bus_lock:
            if self.robot is None:
                return
            try:
                self.robot.bus1.disable_torque(self.robot.left_arm_motors)
            except Exception:
                pass
            try:
                self.robot.disconnect()
            except Exception:
                pass
            self.robot = None
            self.log.info("机械臂已断开")

    def reconnect(self):
        self.log.warning("机械臂重连...")
        try:
            self.disconnect()
        except Exception:
            pass
        time.sleep(1.0)
        self.connect()

    # ---- 读 ----
    def read_joints(self) -> np.ndarray:
        with self._bus_lock:
            if self.robot is None:
                raise RuntimeError("机械臂未连接")
            obs = self.robot.bus1.sync_read("Present_Position", self.robot.left_arm_motors)
            return np.array([float(obs[f"left_arm_{j}"]) for j in JOINTS], dtype=float)

    def read_gripper(self) -> float:
        with self._bus_lock:
            if self.robot is None:
                raise RuntimeError("机械臂未连接")
            obs = self.robot.bus1.sync_read("Present_Position", ["left_arm_gripper"])
            return float(obs["left_arm_gripper"])

    def read_load(self) -> float:
        with self._bus_lock:
            if self.robot is None:
                return 0.0
            for reg in ("Present_Load", "Present_Current"):
                try:
                    obs = self.robot.bus1.sync_read(reg, ["left_arm_gripper"])
                    return abs(float(obs["left_arm_gripper"]))
                except Exception:
                    continue
            return 0.0

    def read_temp(self) -> float:
        with self._bus_lock:
            if self.robot is None:
                return 0.0
            try:
                obs = self.robot.bus1.sync_read("Present_Temperature", self.robot.left_arm_motors)
                vals = []
                for v in obs.values():
                    try:
                        f = float(v)
                    except Exception:
                        continue
                    # STS3215 正常范围 0-100°C；偶发 150/255 是总线/寄存器异常值
                    if 0.0 <= f <= 100.0:
                        vals.append(f)
                return max(vals) if vals else 0.0
            except Exception:
                return 0.0

    # ---- 写 ----
    def send(self, joints, gripper=None, force=False):
        if self.stop_event.is_set() and not force:
            raise StopRequested()
        with self._bus_lock:
            if self.robot is None:
                raise RuntimeError("机械臂未连接")
            if gripper is None:
                gripper = self.gripper
            self.gripper = float(gripper)
            joints = np.clip(np.asarray(joints, dtype=float), -100.0, 100.0)
            action = {f"left_arm_{j}.pos": float(v) for j, v in zip(JOINTS, joints)}
            action[GRIPPER_KEY] = float(gripper)
            self.robot.send_action(action)

    def wait_arrival(self, target, tol=1.2, timeout=0.7) -> bool:
        target = np.asarray(target, dtype=float)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.stop_event.is_set():
                raise StopRequested()
            try:
                q = self.read_joints()
            except Exception:
                time.sleep(0.02)
                continue
            if float(np.abs(q - target).max()) <= tol:
                return True
            time.sleep(1.0 / float(self.cfg["control_hz"]))
        return False

    def move_joints(self, target, gripper=None, speed=None, label=""):
        target = np.clip(np.asarray(target, dtype=float), -100.0, 100.0)
        start = self.read_joints()
        delta = target - start
        max_d = float(np.abs(delta).max())
        if max_d < 0.05:
            self.send(target, gripper)
            return
        speed = float(speed or self.cfg["max_vel_deg_s"])
        # smootherstep 峰值速度 = 1.875 * max_d / T
        T = max(0.18, 1.875 * max_d / max(speed, 1.0))
        T = min(T, 8.0)
        hz = float(self.cfg["control_hz"])
        n = max(2, int(T * hz))
        t0 = time.perf_counter()
        for i in range(1, n + 1):
            if self.stop_event.is_set():
                raise StopRequested()
            s = smootherstep(i / n)
            cmd = start + delta * s
            self.send(cmd, gripper)
            next_t = t0 + i / hz
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
        arrived = self.wait_arrival(target, tol=2.0, timeout=0.9)
        if not arrived:
            q = self.read_joints()
            residual = float(np.abs(q - target).max())
            if residual > 5.0:
                self.log.warning("%s 未完全到位（目标 %s，实际 %s，残差 %.1f）", label,
                                 np.round(target, 1), np.round(q, 1), residual)
            else:
                self.log.debug("%s 到位残差 %.1f（可接受）", label, residual)
        else:
            self.log.debug("%s 完成 %.2fs", label, time.perf_counter() - t0)

    def set_gripper(self, value, settle=None, force=False) -> float:
        q = self.read_joints()
        self.send(q, value, force=force)
        time.sleep(float(settle if settle is not None else self.cfg["gripper_settle_s"]))
        return self.read_gripper()

    def enable_torque(self):
        with self._bus_lock:
            if self.robot is not None:
                self.robot.bus1.enable_torque(self.robot.left_arm_motors)

    def disable_torque(self):
        with self._bus_lock:
            if self.robot is not None:
                try:
                    self.robot.bus1.disable_torque(self.robot.left_arm_motors)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 循环状态机
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, cfg: dict, robot: Robot, camera: CameraStream,
                 detector: Detector, stop_event: threading.Event,
                 logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.robot = robot
        self.camera = camera
        self.det = detector
        self.stop_event = stop_event
        self.log = logger or logging.getLogger("s2b")

        self.table_hover = require_pose(cfg, "hover_joints")
        self.table_pick = require_pose(cfg, "grasp_joints")
        # 标定的抓取深度可能压到桌面；抬高 offset（正数=更高）
        self.table_pick[1] -= float(cfg["grasp_height_offset_deg"])
        self.box_hover = require_pose(cfg, "box_hover_joints")
        self.box_place = require_pose(cfg, "box_drop_joints")

        self.table_drop = self._checked_extra(
            cfg.get("table_drop_joints"), self.table_pick, 10.0,
            lambda: self._offset(self.table_pick, 1, float(cfg["table_drop_lift_deg"])),
            "table_drop_joints")
        # 盆内抓出是独立策略：优先用单独标定的 box_look/box_pick，
        # 不再从 box_drop（放入盆内的释放位）推导。
        if cfg.get("box_pick_joints"):
            self.box_pick = np.asarray(cfg["box_pick_joints"], float)
        else:
            self.box_pick = self._offset(self.box_place, 1, float(cfg["box_pick_deepen_deg"]))
        self.box_pick[1] -= float(cfg["box_pick_height_offset_deg"])
        if cfg.get("box_look_joints"):
            self.box_look = np.asarray(cfg["box_look_joints"], float)
        else:
            self.box_look = self._offset(self.box_hover, 1, -4.0, 3, 4.0)
        self.box_look_pixel = (np.asarray(cfg["box_look_pixel"], float)
                               if cfg.get("box_look_pixel") else None)
        self.box_pick_pixel = (np.asarray(cfg["box_pick_pixel"], float)
                               if cfg.get("box_pick_pixel") else None)
        if cfg.get("safe_joints"):
            self.measure_pose = np.asarray(cfg["safe_joints"], float)
        else:
            # 在 hover 位横向偏转并抬高，离开方块/盒子/桌面，让夹爪自由闭合
            self.measure_pose = self.table_hover.copy()
            self.measure_pose[0] += float(cfg["measure_pan_offset"])
            self.measure_pose[1] -= 10.0
        self.home = (np.asarray(cfg["home_joints"], float)
                     if cfg.get("home_joints") else self.measure_pose.copy())

        self.grasp_pixel = np.asarray(cfg["grasp_pixel"], float)
        self.depth_pixel = np.asarray(cfg["grasp_depth_pixel"], float)
        self.table_band_y = float(cfg.get("table_band_y", 330))

        self.servo_joints = list(cfg["servo_joints"])
        self.servo_idx = [JOINTS.index(j) for j in self.servo_joints]
        self.gain = np.asarray(cfg["gain_matrix"], float)
        if self.gain.ndim != 2 or self.gain.shape[0] != 2 or self.gain.shape[1] != len(self.servo_idx):
            raise SystemExit(
                f"gain_matrix 形状 {self.gain.shape} 与 servo_joints {self.servo_joints} 不匹配")
        self.gain_pinv = np.linalg.pinv(self.gain)
        self.depth_idx = [JOINTS.index(j) for j in
                          ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")]
        dg = cfg.get("depth_gain_matrix")
        self.depth_gain = np.asarray(dg, float) if dg else None
        if self.depth_gain is not None and self.depth_gain.shape != (2, len(self.depth_idx)):
            raise SystemExit(
                f"depth_gain_matrix 形状 {self.depth_gain.shape} 应为 (2, {len(self.depth_idx)})")
        self.roll_slope = cfg.get("roll_slope_deg_per_deg")

        self.air_closed = None
        self.load_available = False
        self.load_thresh = float(cfg["grasp_load_min"])
        self.hold = float(cfg["gripper_open"])

    @staticmethod
    def _offset(base: np.ndarray, idx: int, delta: float,
                idx2: int | None = None, delta2: float = 0.0) -> np.ndarray:
        p = base.copy()
        p[idx] += delta
        if idx2 is not None:
            p[idx2] += delta2
        return p

    def _checked_extra(self, value, ref: np.ndarray, max_delta: float,
                       fallback, name: str) -> np.ndarray:
        """额外位姿只在和基准位接近时采用；否则从基准位推导。

        15:32 那次标定把 box_look/table_drop/home 都录成了折叠位，
        直接使用会让盒内抓取从折叠位跳到伸展位，方块中途丢失。
        """
        if value:
            arr = np.asarray(value, float)
            if arr.shape == ref.shape and float(np.abs(arr - ref).max()) <= max_delta:
                return arr
            self.log.warning("%s 与基准位相差过大，忽略并自动推导", name)
        return fallback()

    # ---- 初始化 ----
    def prepare(self):
        self.robot.connect()
        self.camera.start()
        time.sleep(0.5)
        open_cmd = float(self.cfg["gripper_open"])
        self.robot.set_gripper(open_cmd, settle=0.35)
        self.robot.move_joints(self.measure_pose, open_cmd,
                               speed=35, label="到测量安全位")
        air = self.robot.set_gripper(float(self.cfg["gripper_closed"]), settle=0.80)
        air_load = self.robot.read_load()
        self.air_closed = air
        # Present_Load 在部分固件上会返回 500 之类的饱和值，不可信
        self.load_available = 0.5 < air_load < 300.0
        self.load_thresh = air_load + float(self.cfg["grasp_load_min"])
        self.robot.set_gripper(open_cmd, settle=0.35)
        self.robot.move_joints(self.home, open_cmd, speed=35, label="回安全位")
        self.log.info("测量位=%s", np.round(self.measure_pose, 1))
        self.log.info("空合值=%.2f 空载=%.1f 负载可用=%s 阈值=%.1f",
                      air, air_load, self.load_available, self.load_thresh)
        if air > 20.0:
            self.log.warning("空合值偏高（%.1f），夹爪可能被挡；请检查测量位", air)
        self.log.info("位姿: table_hover=%s table_pick=%s box_hover=%s box_place=%s",
                      np.round(self.table_hover, 1), np.round(self.table_pick, 1),
                      np.round(self.box_hover, 1), np.round(self.box_place, 1))
        self.log.info("派生: table_drop=%s box_pick=%s box_look=%s",
                      np.round(self.table_drop, 1), np.round(self.box_pick, 1),
                      np.round(self.box_look, 1))

    # ---- 检测辅助 ----
    def box_roi(self, wide=False):
        cx, cy = self.cfg["box_target_pixel"]
        r = int(float(self.cfg["box_roi_radius"]) * (2.0 if wide else 1.0))
        h, w = 480, 640
        return (max(0, int(cx - r)), max(0, int(cy - r)),
                min(w, int(cx + r)), min(h, int(cy + r)))

    def frame_and_det(self, roi=None):
        frame = self.camera.read()
        return frame, self.det.detect(frame, roi)

    # ---- 视觉伺服 ----
    def servo_to_cube(self, seed: np.ndarray, target_px: np.ndarray,
                      roi=None, max_iters=None):
        goal = seed.copy()
        pan_idx = JOINTS.index("shoulder_pan")
        lost = 0
        max_iters = int(max_iters or self.cfg["servo_max_iters"])
        for it in range(max_iters):
            if self.stop_event.is_set():
                raise StopRequested()
            frame = self.camera.read()
            det = self.det.detect(frame, roi)
            if det is None:
                lost += 1
                if lost >= 10:
                    raise TargetLost("连续 10 帧看不到方块")
                if lost in (3, 6):
                    for dp in (-6, -3, 3, 6, 0):
                        p = goal.copy()
                        p[pan_idx] += dp
                        self.robot.send(p, float(self.cfg["gripper_open"]))
                        time.sleep(0.16)
                        d = self.det.detect(self.camera.read(), roi)
                        if d is not None:
                            det = d
                            goal = p
                            lost = 0
                            break
                if det is None:
                    time.sleep(0.04)
                    continue
            lost = 0
            err = det.center - target_px
            if float(np.abs(err).max()) < float(self.cfg["servo_tol_px"]):
                self.log.info("伺服收敛: 误差 (%.0f,%.0f)px，%d 次", err[0], err[1], it)
                return goal, det
            delta = self.gain_pinv @ (-err * float(self.cfg["servo_kp"]))
            delta = np.clip(delta, -float(self.cfg["servo_step_deg"]),
                            float(self.cfg["servo_step_deg"]))
            goal[self.servo_idx] += delta
            goal = np.clip(goal, -100.0, 100.0)
            self.robot.send(goal, float(self.cfg["gripper_open"]))
            time.sleep(1.0 / float(self.cfg["servo_hz"]))
        raise ServoTimeout(f"伺服 {max_iters} 次未收敛")

    def detect_jaw_tips(self, frame):
        """检测张开的两指指尖中点；失败返回 None。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, S, V = cv2.split(hsv)
        m = ((V > 215) & (S < 60)).astype(np.uint8) * 255
        m[:300, :] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 1500]
        if len(cnts) < 2:
            return None
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        pts = []
        for c in cnts:
            idx = int(c[:, :, 1].argmin())
            pts.append(c[idx][0])
        (x1, y1), (x2, y2) = pts
        if abs(x1 - x2) < 25 or abs(x1 - x2) > 560:
            return None
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def depth_target(self) -> np.ndarray:
        """深度对准目标：优先用现场指隙 + grasp_dy_px，失败用标定值。"""
        try:
            gap = self.detect_jaw_tips(self.camera.read())
        except CameraError:
            gap = None
        if gap is not None:
            t = np.array([gap[0], gap[1] + float(self.cfg.get("grasp_dy_px", 80.0))])
            self.log.info("指隙实测 (%.0f,%.0f) -> 深度目标 (%.0f,%.0f)",
                          gap[0], gap[1], t[0], t[1])
            return t
        return self.depth_pixel

    def depth_align(self, goal: np.ndarray) -> Optional[np.ndarray]:
        """深度处用预标定的 2x4 雅可比修正；没有标定就现场探测一次。"""
        if self.depth_gain is None:
            return self.depth_align_probe(goal)
        pinv = np.linalg.pinv(self.depth_gain)
        goal = goal.copy()
        target = self.depth_target()
        det = None
        for it in range(int(self.cfg["depth_max_iters"])):
            if self.stop_event.is_set():
                raise StopRequested()
            frame = self.camera.read()
            det = self.det.detect(frame)
            if det is None:
                self.log.warning("深度对准中丢失方块")
                return None
            err = det.center - target
            if float(np.abs(err).max()) < float(self.cfg["depth_tol_px"]):
                break
            delta = pinv @ (-err * float(self.cfg["depth_gain"]))
            delta = np.clip(delta, -float(self.cfg["depth_step_deg"]),
                            float(self.cfg["depth_step_deg"]))
            goal[self.depth_idx] += delta
            goal = np.clip(goal, -100.0, 100.0)
            self.robot.send(goal, float(self.cfg["gripper_open"]))
            time.sleep(0.10)
        if det is None:
            return None
        err = det.center - target
        if float(np.abs(err).max()) > 25.0:
            self.log.warning("深度对准偏差过大 (%.0f,%.0f)px，放弃", err[0], err[1])
            return None
        return goal

    def depth_align_probe(self, goal: np.ndarray) -> Optional[np.ndarray]:
        """没有 depth_gain_matrix 时，在抓取深度现场探测 2x4 雅可比。

        只在第一次抓取时做一次，成功后写回 config.json，后续直接复用。
        """
        self.log.info("深度雅可比未标定，现场探测（约 2-4s）")
        open_cmd = float(self.cfg["gripper_open"])
        base = goal.copy()
        target = self.depth_target()
        J = np.zeros((2, len(self.depth_idx)))
        for col, idx in enumerate(self.depth_idx):
            p = base.copy()
            p[idx] += 2.0
            self.robot.send(p, open_cmd)
            time.sleep(0.20)
            d1 = self.det.detect(self.camera.read())
            p[idx] -= 4.0
            self.robot.send(p, open_cmd)
            time.sleep(0.20)
            d2 = self.det.detect(self.camera.read())
            self.robot.send(base, open_cmd)
            time.sleep(0.18)
            if d1 is None or d2 is None:
                self.log.warning("探测 %s 时丢失方块", JOINTS[idx])
                continue
            J[:, col] = [(d1.cx - d2.cx) / 4.0, (d1.cy - d2.cy) / 4.0]
            self.log.info("深度探测 %s: 像素/度 = (%.2f, %.2f)",
                          JOINTS[idx], J[0, col], J[1, col])
        if float(np.abs(J).max()) < 1.0:
            self.log.warning("深度探测响应过小，跳过修正")
            return base
        self.depth_gain = J
        self.cfg["depth_gain_matrix"] = J.tolist()
        save_config({"depth_gain_matrix": J.tolist()})
        pinv = np.linalg.pinv(J)
        det = None
        for it in range(6):
            frame = self.camera.read()
            det = self.det.detect(frame)
            if det is None:
                return None
            err = det.center - target
            if float(np.abs(err).max()) < float(self.cfg["depth_tol_px"]):
                break
            delta = np.clip(pinv @ (-err * 0.8),
                            -float(self.cfg["depth_step_deg"]),
                            float(self.cfg["depth_step_deg"]))
            base[self.depth_idx] += delta
            self.robot.send(base, open_cmd)
            time.sleep(0.12)
        if det is None:
            return None
        err = det.center - target
        if float(np.abs(err).max()) > 25.0:
            self.log.warning("现场探测后仍偏差 (%.0f,%.0f)px", err[0], err[1])
            return None
        return base

    def align_roll(self, goal: np.ndarray) -> np.ndarray:
        roll = JOINTS.index("wrist_roll")
        frame = self.camera.read()
        det = self.det.detect(frame)
        if det is None or abs(det.angle) < 10.0:
            return goal
        goal = goal.copy()
        slope = self.roll_slope
        if slope is None:
            q = goal.copy()
            q[roll] += 8.0
            self.robot.send(q, float(self.cfg["gripper_open"]))
            time.sleep(0.30)
            det1 = self.det.detect(self.camera.read())
            self.robot.send(goal, float(self.cfg["gripper_open"]))
            time.sleep(0.18)
            if det1 is None:
                return goal
            slope = (det1.angle - det.angle) / 8.0
            if abs(slope) < 0.05:
                return goal
            self.roll_slope = slope
            self.cfg["roll_slope_deg_per_deg"] = slope
            save_config({"roll_slope_deg_per_deg": slope})
            self.log.info("腕部角度斜率标定: %.3f deg/deg", slope)
        corr = clamp(-det.angle / slope, -30.0, 30.0)
        q = goal.copy()
        q[roll] += corr
        self.robot.send(q, float(self.cfg["gripper_open"]))
        time.sleep(0.28)
        det2 = self.det.detect(self.camera.read())
        if det2 is not None and abs(det2.angle) < abs(det.angle):
            self.log.info("转腕对准: %.0f° -> %.0f°", det.angle, det2.angle)
            return q
        self.robot.send(goal, float(self.cfg["gripper_open"]))
        time.sleep(0.15)
        return goal

    # ---- 抓取/验证 ----
    def grasp_attempt(self, goal: np.ndarray, box: bool) -> bool:
        close_cmd = float(self.cfg["gripper_closed"])
        pos = self.robot.set_gripper(close_cmd, settle=float(self.cfg["gripper_settle_s"]))
        load = self.robot.read_load()
        by_pos = pos > float(self.air_closed) + float(self.cfg["grasp_pos_margin"])
        by_load = self.load_available and load >= self.load_thresh
        self.log.info("闭爪: pos=%.2f (空合 %.2f) load=%.1f -> %s",
                      pos, float(self.air_closed), load,
                      "夹住" if (by_pos or by_load) else "空爪")
        if not (by_pos or by_load):
            return False
        # 搬运期间保持全夹持力；之前用 pos-2 会松开夹爪，盒内抓取容易滑脱
        self.hold = close_cmd
        lift = goal.copy()
        lift[1] -= 4.0  # shoulder_lift 减小 = 抬高
        self.robot.move_joints(lift, self.hold, speed=30, label="预抬升")
        # 盆内抓取抬升到盆口上方，而不是跳到 box_look（可能是很远的观察位）
        hover = self.box_look if box else self.table_hover
        self.robot.move_joints(hover, self.hold, speed=45, label="抬回观察位")
        if box:
            # 盒内方块和夹爪里的方块在画面里很难区分；用抬升后的
            # 位置/负载是否保持来验证，避免“闭爪瞬间有负载、抬起来却滑掉”。
            pos2 = self.robot.read_gripper()
            load2 = self.robot.read_load()
            held_pos = pos2 > float(self.air_closed) + float(self.cfg["grasp_pos_margin"])
            held_load = self.load_available and load2 >= self.load_thresh
            self.log.info("盒内抬升后: pos=%.2f load=%.1f -> %s",
                          pos2, load2, "夹住" if (held_pos or held_load) else "滑脱")
            if held_pos or held_load:
                return True
            self.log.warning("盒内抓取抬升后掉落/滑脱，判定未夹住")
            return False
        frame = self.camera.read()
        det = self.det.detect(frame)
        if det is not None and float(np.linalg.norm(det.center - self.grasp_pixel)) < 60.0:
            self.log.warning("抬升后方块仍在抓取点附近，判定未夹住")
            return False
        self.log.info("抓取验证通过")
        return True

    def verify_in_box(self) -> bool:
        # place_in_box 抬离后已经在 box_hover；用盆口视角验证，
        # 不再跳到 box_look（那是盆内抓取的观察位，容易漏检）。
        frame = self.camera.read()
        det_box = self.det.detect(frame, self.box_roi())
        if det_box is not None:
            self.log.info("盆内检测到方块")
            return True
        det_all = self.det.detect(frame)
        if det_all is not None and not in_roi(det_all.center, self.box_roi()):
            self.log.warning("盆内没看到，盆外看到方块 -> 放入失败")
            return False
        p = self.box_hover.copy()
        p[1] -= 4.0
        self.robot.move_joints(p, float(self.cfg["gripper_open"]), speed=20, label="抬高观察盆内")
        det2 = self.det.detect(self.camera.read(), self.box_roi())
        if det2 is not None:
            self.log.info("抬高后盆内检测到方块")
            return True
        self.log.error("盆内方块不可见，无法确认放入成功")
        return False

    def verify_on_table(self) -> bool:
        frame = self.camera.read()
        det = self.det.detect(frame)
        if det is not None and det.cy > 240:
            self.log.info("桌面检测到方块")
            return True
        p = self.table_hover.copy()
        p[1] -= 3.0
        self.robot.move_joints(p, float(self.cfg["gripper_open"]), speed=20, label="抬高确认")
        det2 = self.det.detect(self.camera.read())
        if det2 is not None:
            self.log.info("抬高后桌面检测到方块")
            return True
        self.log.warning("桌面没看到方块，按放置成功处理（下一轮会验证）")
        return True

    # ---- 四个阶段 ----
    def scan_for_cube(self):
        """在桌面上做 pan/wrist 扫描，找到后伺服对中。返回 (pose, det) 或 None。"""
        open_cmd = float(self.cfg["gripper_open"])
        base = self.table_hover.copy()
        pan = JOINTS.index("shoulder_pan")
        wf = JOINTS.index("wrist_flex")
        for dp in (0, -8, 8, -16, 16):
            for dw in (0, -6, 6):
                p = base.copy()
                p[pan] += dp
                p[wf] += dw
                self.robot.move_joints(p, open_cmd, speed=35, label="扫描桌面")
                time.sleep(0.12)
                det = self.det.detect(self.camera.read())
                if det is not None:
                    self.log.info("扫描 pan%+d wrist%+d 找到方块", dp, dw)
                    try:
                        return self.servo_to_cube(p, self.grasp_pixel)
                    except (TargetLost, ServoTimeout):
                        continue
        return None

    def servo_or_scan(self):
        """桌面伺服；失败则扫描。返回 (pose, det) 或 None。"""
        try:
            return self.servo_to_cube(self.table_hover, self.grasp_pixel)
        except (TargetLost, ServoTimeout) as e:
            self.log.warning("桌面伺服失败（%s），开始扫描", e)
            return self.scan_for_cube()

    def pick_from_table(self) -> bool:
        open_cmd = float(self.cfg["gripper_open"])
        self.log.info("阶段: 桌面抓取")
        self.robot.move_joints(self.table_hover, open_cmd, speed=45, label="去桌面上方")
        found = self.servo_or_scan()
        if found is None:
            self.log.error("扫描后仍找不到方块")
            return False
        goal, det = found
        goal = self.align_roll(goal)
        delta = self.table_pick - self.table_hover
        grasp_goal = goal + delta
        deepen = min(float(self.cfg.get("deepen_per_retry_deg", 3.0)), 1.0)
        for attempt in range(1, int(self.cfg["grasp_retries"]) + 1):
            self.log.info("桌面抓取尝试 %d/%d", attempt, self.cfg["grasp_retries"])
            self.robot.move_joints(grasp_goal, open_cmd, speed=22, label="下探")
            depth_goal = self.depth_align(grasp_goal)
            if depth_goal is not None:
                depth_goal = depth_goal.copy()
                # 深度修正不允许把夹爪压回桌面以下（最多再低 1 个单位）
                depth_goal[1] = min(depth_goal[1], self.table_pick[1] + 1.0)
            if depth_goal is not None and self.grasp_attempt(depth_goal, box=False):
                return True
            self.robot.set_gripper(open_cmd, settle=0.30)
            self.robot.move_joints(self.table_hover, open_cmd, speed=40, label="回悬停")
            found = self.servo_or_scan()
            if found is None:
                self.log.error("重试时找不到方块")
                return False
            goal, det = found
            goal = self.align_roll(goal)
            grasp_goal = goal + delta
            grasp_goal[1] += deepen * attempt
        self.log.error("桌面抓取 %d 次都失败", self.cfg["grasp_retries"])
        return False

    def place_in_box(self) -> bool:
        self.log.info("阶段: 放入盒子")
        self.robot.move_joints(self.box_hover, self.hold, speed=45, label="去盒子上方")
        self.robot.move_joints(self.box_place, self.hold, speed=18, label="下降入盒")
        self.robot.set_gripper(float(self.cfg["gripper_open"]), settle=0.35)
        self.robot.move_joints(self.box_hover, float(self.cfg["gripper_open"]),
                               speed=25, label="抬离盒子")
        return self.verify_in_box()

    def box_servo(self):
        """盆内观察位对中：只有单独标定了 box_look_pixel 才做视觉对中。

        没有 box_look_pixel 时直接用标定的 box_look/box_pick 位姿对，
        避免用桌面的 grasp_pixel 把夹爪带偏。
        """
        if self.box_look_pixel is None:
            return self.box_look.copy(), self.det.detect(self.camera.read())
        try:
            return self.servo_to_cube(self.box_look, self.box_look_pixel,
                                      roi=self.box_roi())
        except (TargetLost, ServoTimeout):
            self.log.warning("盆内观察位对中失败，放宽 ROI")
            return self.servo_to_cube(self.box_look, self.box_look_pixel,
                                      roi=self.box_roi(wide=True))

    def pick_from_box(self) -> bool:
        """盆内抓出：独立策略，不复用放入盆内的 box_hover/box_drop 推导。

        - box_look/box_pick 优先使用单独标定值；
        - box_pick_height_offset_deg 可整体抬高，避免夹爪压盆底；
        - 重试时略微抬高（不是像桌面抓取那样下探）；
        - 抬升后复检位置/负载，滑脱判失败。
        """
        open_cmd = float(self.cfg["gripper_open"])
        self.log.info("阶段: 从盆内抓出（独立策略）")
        self.robot.move_joints(self.box_look, open_cmd, speed=35, label="到盆内观察位")
        # 腕相机固定在夹爪上：方块夹在两指之间的像素位置是固定的，
        # 就是桌面标定的 grasp_depth_pixel。盆内抓取位也用它做深度对准。
        # 实测：对准后闭爪 pos=5.92/load=256，抬到盆口 pos=5.47/load=200，成功取出。
        grasp_goal = self.box_pick.copy()
        for attempt in range(1, int(self.cfg["grasp_retries"]) + 1):
            self.log.info("盆内抓取尝试 %d/%d", attempt, self.cfg["grasp_retries"])
            self.robot.move_joints(grasp_goal, open_cmd, speed=12, label="盆内下探")
            aligned = self.depth_align(grasp_goal)
            if aligned is not None:
                grasp_goal = aligned
            d = self.det.detect(self.camera.read())
            if d is None:
                self.log.warning("盆内抓取位看不到方块，重试")
            elif self.grasp_attempt(grasp_goal, box=True):
                return True
            self.robot.set_gripper(open_cmd, settle=0.30)
            self.robot.move_joints(self.box_look, open_cmd, speed=30, label="回盆内观察位")
            # 盆内重试向上抬，避免越试越压盆底
            grasp_goal = self.box_pick.copy()
            grasp_goal[1] -= 1.0 * attempt
        self.log.error("盆内抓取 %d 次都失败", self.cfg["grasp_retries"])
        return False

    def place_on_table(self) -> bool:
        self.log.info("阶段: 放回桌面")
        self.robot.move_joints(self.table_hover, self.hold, speed=45, label="回桌面上方")
        self.robot.move_joints(self.table_drop, self.hold, speed=18, label="下降放桌面")
        self.robot.set_gripper(float(self.cfg["gripper_open"]), settle=0.35)
        self.robot.move_joints(self.table_hover, float(self.cfg["gripper_open"]),
                               speed=25, label="抬离桌面")
        return self.verify_on_table()

    # ---- 单轮 ----
    def run_cycle(self):
        t0 = time.time()
        if not self.pick_from_table():
            raise CycleError("桌面抓取失败")
        if not self.place_in_box():
            raise CycleError("放入盒子失败")
        if not self.pick_from_box():
            raise CycleError("盒内抓取失败")
        if not self.place_on_table():
            raise CycleError("放回桌面失败")
        self.log.info("本轮完成，用时 %.1fs", time.time() - t0)

    # ---- 恢复 ----
    def recover(self):
        self.log.warning("执行恢复流程")
        open_cmd = float(self.cfg["gripper_open"])
        try:
            self.robot.set_gripper(open_cmd, settle=0.30)
        except Exception:
            pass
        if self.camera.age() > 2.0:
            self.camera.reconnect()
            time.sleep(0.5)
        # 如果方块被留在盒子里，先取回桌面，避免下一轮一直找不到
        try:
            self.robot.move_joints(self.box_look, open_cmd, speed=30, label="恢复观察盒子")
            det = self.det.detect(self.camera.read(), self.box_roi())
            if det is not None:
                self.log.info("恢复：盒内发现方块，尝试取回桌面")
                if self.pick_from_box():
                    self.place_on_table()
        except StopRequested:
            raise
        except Exception as e:
            self.log.warning("恢复时盒内取回失败: %s", e)
        try:
            self.robot.move_joints(self.home, open_cmd, speed=30, label="恢复回安全位")
        except StopRequested:
            raise
        except Exception as e:
            self.log.error("恢复回安全位失败: %s", e)
        time.sleep(float(self.cfg["recover_wait_s"]))

    def shutdown(self):
        try:
            self.robot.set_gripper(float(self.cfg["gripper_open"]), settle=0.10, force=True)
        except Exception:
            pass
        self.robot.disable_torque()
        self.robot.disconnect()
        self.camera.stop()


# ---------------------------------------------------------------------------
# 日志/看门狗/常驻
# ---------------------------------------------------------------------------

def setup_logging(cfg: dict) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("s2b")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
    fh = RotatingFileHandler(
        LOG_DIR / "square2box_loop.log",
        maxBytes=int(cfg["log_max_mb"]) * 1024 * 1024,
        backupCount=int(cfg["log_backups"]),
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class CycleCSV:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header = ["ts", "cycle", "success", "duration_s", "phase", "error"]

    def write(self, cycle: int, success: bool, duration: float, phase: str, error: str):
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(self._header)
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), cycle,
                        int(success), round(duration, 2), phase, error])


class Supervisor:
    def __init__(self, cfg: dict, logger: logging.Logger):
        self.cfg = cfg
        self.log = logger
        self.stop_event = threading.Event()
        self.thermal_pause = threading.Event()
        self._hot_streak = 0
        self.robot = Robot(cfg, self.stop_event, logger)
        self.camera = CameraStream(device=camera_device(cfg), logger=logger)
        self.detector = Detector(cfg, logger)
        self.runner = Runner(cfg, self.robot, self.camera, self.detector,
                             self.stop_event, logger)
        self.csv = CycleCSV(LOG_DIR / "cycles.csv")
        self._watchdog = None

    def _install_signals(self):
        def handler(signum, frame):
            self.log.warning("收到信号 %s，准备停止", signum)
            self.stop_event.set()
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _watchdog_loop(self):
        while not self.stop_event.wait(1.0):
            try:
                age = self.camera.age()
                if age > 3.0:
                    self.log.warning("相机 %.1fs 无新帧，触发重连", age)
                    self.camera.reconnect()
                temp = self.robot.read_temp()
                limit = float(self.cfg["temp_limit_c"])
                if temp > limit:
                    self._hot_streak += 1
                    self.log.warning("电机温度 %.0f°C 超限 (%d/2)", temp, self._hot_streak)
                    if self._hot_streak >= 2 and not self.thermal_pause.is_set():
                        self.log.error("电机温度连续超限，卸力等待")
                        self.thermal_pause.set()
                        self.robot.disable_torque()
                else:
                    self._hot_streak = 0
            except Exception as e:
                self.log.debug("看门狗异常: %s", e)

    def run(self, max_cycles: int = 0, max_seconds: float = 0):
        self._install_signals()
        while not self.stop_event.is_set():
            try:
                self.runner.prepare()
                break
            except StopRequested:
                try:
                    self.runner.shutdown()
                except Exception:
                    pass
                return
            except Exception as e:
                self.log.exception("初始化失败，5s 后重试: %s", e)
                try:
                    self.runner.shutdown()
                except Exception:
                    pass
                time.sleep(5)
        if self.stop_event.is_set():
            try:
                self.runner.shutdown()
            except Exception:
                pass
            return
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True,
                                          name="watchdog")
        self._watchdog.start()
        t_start = time.time()
        cycle = 0
        failures = 0
        success = 0
        try:
            while not self.stop_event.is_set():
                if max_seconds and time.time() - t_start >= max_seconds:
                    self.log.info("达到 max_seconds=%.0f，停止", max_seconds)
                    break
                if max_cycles and cycle >= max_cycles:
                    break
                if self.thermal_pause.is_set():
                    self.log.warning("温度保护中，等待 %.0fs", float(self.cfg["temp_wait_s"]))
                    time.sleep(float(self.cfg["temp_wait_s"]))
                    self.robot.enable_torque()
                    try:
                        self.runner.prepare()
                    except Exception as e:
                        self.log.error("温度恢复失败: %s", e)
                    self.thermal_pause.clear()
                # 每轮开始前直接查一次温度，避免热机启动后中途被看门狗打断
                temp = self.robot.read_temp()
                if temp > float(self.cfg["temp_limit_c"]):
                    wait = float(self.cfg["temp_wait_s"])
                    self.log.warning("启动前温度 %.0f°C 超限，先冷却 %.0fs", temp, wait)
                    self.robot.disable_torque()
                    time.sleep(wait)
                    self.robot.enable_torque()
                    try:
                        self.runner.prepare()
                    except Exception as e:
                        self.log.error("冷却后恢复失败: %s", e)
                cycle += 1
                t0 = time.time()
                phase = "cycle"
                err = ""
                try:
                    self.runner.run_cycle()
                    success += 1
                    failures = 0
                    ok = True
                except StopRequested:
                    self.log.warning("用户停止")
                    break
                except Exception as e:
                    failures += 1
                    ok = False
                    err = f"{type(e).__name__}: {e}"
                    self.log.exception("第 %d 轮失败: %s", cycle, err)
                    phase = "recover"
                    if self.thermal_pause.is_set():
                        self.log.warning("温度保护中，跳过本轮恢复")
                    else:
                        try:
                            self.runner.recover()
                        except StopRequested:
                            break
                        except Exception as e2:
                            self.log.error("恢复失败: %s", e2)
                            try:
                                self.robot.reconnect()
                                self.runner.prepare()
                            except Exception as e3:
                                self.log.error("重连失败: %s", e3)
                                time.sleep(5)
                duration = time.time() - t0
                self.csv.write(cycle, ok, duration, phase, err)
                self.log.info("统计: 成功 %d / 失败 %d / 连续失败 %d，本轮 %.1fs",
                              success, failures, failures, duration)
                if failures >= int(self.cfg["max_consecutive_failures"]):
                    self.log.error("连续失败 %d 次，执行硬恢复", failures)
                    try:
                        self.robot.disable_torque()
                        time.sleep(5)
                        self.robot.connect()
                        self.runner.prepare()
                        failures = 0
                    except Exception as e:
                        self.log.error("硬恢复失败: %s", e)
                        time.sleep(10)
                time.sleep(float(self.cfg["cycle_pause_s"]))
        finally:
            self.log.info("退出清理...")
            self.runner.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_self_test(cfg: dict, logger: logging.Logger):
    print("=== 配置检查 ===")
    for k in ("hover_joints", "grasp_joints", "box_hover_joints", "box_drop_joints",
              "grasp_pixel", "grasp_depth_pixel", "gain_matrix"):
        print(f"  {k}: {'OK' if cfg.get(k) is not None else 'MISSING'}")
    print(f"  servo_joints: {cfg['servo_joints']}")
    print(f"  gain_matrix shape: {np.asarray(cfg['gain_matrix']).shape}")
    print(f"  depth_gain_matrix: {'OK' if cfg.get('depth_gain_matrix') else '未标定（可跳过）'}")

    print("\n=== 相机 ===")
    cam = CameraStream(device=camera_device(cfg), logger=logger)
    cam.start()
    try:
        time.sleep(0.8)
        for _ in range(5):
            f = cam.read()
        print(f"  画面 {f.shape}, fps≈{cam.fps_est:.1f}")
        det = Detector(cfg, logger)
        d = det.detect(f)
        print(f"  YOLO: {'已加载' if det.model is not None else '未加载（HSV 兜底）'}")
        print(f"  检测: {d}")
        cv2.imwrite(str(LOG_DIR / "self_test_frame.jpg"), det.annotate(f.copy(), d))
    finally:
        cam.stop()

    print("\n=== 机械臂（只读） ===")
    robot = Robot(cfg, threading.Event(), logger)
    try:
        robot.connect()
        q = robot.read_joints()
        print(f"  关节: {np.round(q, 2).tolist()}")
        print(f"  夹爪: {robot.read_gripper():.2f}")
        print(f"  负载: {robot.read_load():.1f}")
        print(f"  温度: {robot.read_temp():.0f}°C")
    finally:
        robot.disconnect()
    print("\nself-test 完成（未动电机）")


def cmd_preview(cfg: dict, logger: logging.Logger, seconds=10.0, display=False,
                hover=False, box=False):
    robot = None
    if hover or box:
        robot = Robot(cfg, threading.Event(), logger)
        robot.connect()
        robot.set_gripper(float(cfg["gripper_open"]), settle=0.2)
        pose = require_pose(cfg, "box_hover_joints") if box else require_pose(cfg, "hover_joints")
        robot.move_joints(pose, float(cfg["gripper_open"]), speed=30,
                          label="preview 盒子上方" if box else "preview 悬停位")
    cam = CameraStream(device=camera_device(cfg), logger=logger)
    det = Detector(cfg, logger)
    cam.start()
    t0 = time.time()
    last_print = 0.0
    try:
        while time.time() - t0 < seconds:
            try:
                frame = cam.read()
            except CameraError as e:
                print("相机:", e)
                time.sleep(0.2)
                continue
            d = det.detect(frame)
            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                if d is None:
                    print(f"[{now - t0:5.1f}s] NO TARGET")
                else:
                    print(f"[{now - t0:5.1f}s] {d.source:8s} conf={d.conf:.2f} "
                          f"center=({d.cx:.0f},{d.cy:.0f}) area={d.area:.0f} "
                          f"angle={d.angle:.1f}")
            if display:
                vis = det.annotate(frame.copy(), d)
                cv2.imshow("square2box preview", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cam.stop()
        if robot is not None:
            robot.disconnect()
        if display:
            cv2.destroyAllWindows()


def cmd_calibrate_extra(cfg: dict, logger: logging.Logger):
    robot = Robot(cfg, threading.Event(), logger)
    try:
        robot.connect()
        robot.disable_torque()
        time.sleep(0.3)
        print("\n== 额外位姿标定（全程卸力，用手摆臂）==")
        prompts = [
            ("box_pick_joints", "把夹爪套在【盒子里的方块】上（抓取深度）"),
            ("table_drop_joints", "把夹爪降到【桌面放方块】的释放高度"),
            ("box_look_joints", "摆到能看清【盒子内部】的观察位"),
            ("home_joints", "摆到【安全休息位】（远离桌面和盒子）"),
        ]
        for key, text in prompts:
            input(f"{text}，按回车记录 {key} ... ")
            cfg[key] = robot.read_joints().tolist()
            print(f"  {key} = {np.round(cfg[key], 2).tolist()}")
        save_config({k: cfg[k] for k, _ in prompts})
        print("额外位姿已保存到 config.json")
    finally:
        robot.disconnect()


def cmd_calibrate_depth(cfg: dict, logger: logging.Logger):
    robot = Robot(cfg, threading.Event(), logger)
    cam = CameraStream(device=camera_device(cfg), logger=logger)
    det = Detector(cfg, logger)
    try:
        robot.connect()
        cam.start()
        time.sleep(0.8)
        robot.set_gripper(float(cfg["gripper_open"]), settle=0.35)
        base = require_pose(cfg, "grasp_joints")
        base[1] -= float(cfg["grasp_height_offset_deg"])
        robot.move_joints(base, float(cfg["gripper_open"]), speed=20, label="到抬高后的抓取深度")
        time.sleep(0.4)
        d0 = det.detect(cam.read())
        if d0 is None:
            raise SystemExit("抓取深度看不到方块，先调好位置/HSV 再试")
        cfg["grasp_depth_pixel"] = [float(d0.cx), float(d0.cy)]
        save_config({"grasp_depth_pixel": cfg["grasp_depth_pixel"]})
        print(f"深度目标像素已保存: ({d0.cx:.1f}, {d0.cy:.1f})")
        idxs = [JOINTS.index(j) for j in
                ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")]
        J = np.zeros((2, len(idxs)))
        for col, idx in enumerate(idxs):
            p = base.copy()
            p[idx] += 2.0
            robot.move_joints(p, float(cfg["gripper_open"]), speed=8)
            time.sleep(0.25)
            d1 = det.detect(cam.read())
            p[idx] -= 4.0
            robot.move_joints(p, float(cfg["gripper_open"]), speed=8)
            time.sleep(0.25)
            d2 = det.detect(cam.read())
            robot.move_joints(base, float(cfg["gripper_open"]), speed=8)
            time.sleep(0.2)
            if d1 is None or d2 is None:
                print(f"  {JOINTS[idx]}: 探测丢失，跳过")
                continue
            J[:, col] = [(d1.cx - d2.cx) / 4.0, (d1.cy - d2.cy) / 4.0]
            print(f"  {JOINTS[idx]}: 像素/度 = ({J[0, col]:.2f}, {J[1, col]:.2f})")
        if np.abs(J).max() < 1.0:
            raise SystemExit("探测响应过小，深度雅可比不可用")
        cfg["depth_gain_matrix"] = J.tolist()
        save_config({"depth_gain_matrix": J.tolist()})
        print("深度雅可比已保存")
    finally:
        cam.stop()
        robot.disconnect()


def cmd_calibrate_box_pick(cfg: dict, logger: logging.Logger, test: bool = False):
    """单独标定盆内抓出：box_look/box_pick + 对应像素。

    与“放入盆内”的 box_hover/box_drop 解耦；抓取位可以按需要抬高，
    避免夹爪压到盆底。
    """
    robot = Robot(cfg, threading.Event(), logger)
    cam = CameraStream(device=camera_device(cfg), logger=logger)
    det = Detector(cfg, logger)
    try:
        robot.connect()
        cam.start()
        time.sleep(0.8)
        robot.disable_torque()
        time.sleep(0.3)
        print("\n== 盆内抓出单独标定（全程卸力，用手摆臂）==")
        print("目标：从盒口观察位能稳定看到方块；抓取位中方块应位于两指之间，"
              "且检测框不得被指爪/盆沿大面积遮住。不要压到盆底。")

        input("1/3 摆到能看清盆内方块的观察位，按回车记录 box_look ... ")
        look = robot.read_joints().tolist()
        cfg["box_look_joints"] = look
        d0 = det.detect(cam.read())
        if d0 is not None:
            cfg["box_look_pixel"] = [float(d0.cx), float(d0.cy)]
            print(f"  box_look_pixel = ({d0.cx:.1f}, {d0.cy:.1f})")
        else:
            cfg["box_look_pixel"] = None
            print("  观察位没检测到方块，box_look_pixel=None（跳过观察位对中）")

        input("2/3 把夹爪降到两指能从方块两侧包住、但不碰盆底的位置，按回车记录 box_pick ... ")
        pick = robot.read_joints().tolist()
        frame = cam.read()
        d1 = det.detect(frame)
        cv2.imwrite(str(LOG_DIR / "box_pick_calibration.jpg"), det.annotate(frame.copy(), d1))
        if d1 is None:
            raise SystemExit("抓取位看不到方块，未保存；请抬高/调整夹爪，确保目标可见后重试")
        if d1.area > 6000:
            raise SystemExit(
                f"抓取位检测框异常大（{d1.area:.0f}px）；方块可能被夹爪/盆沿遮挡，未保存")
        cfg["box_pick_joints"] = pick
        cfg["box_pick_pixel"] = [float(d1.cx), float(d1.cy)]
        print(f"  box_pick_pixel = ({d1.cx:.1f}, {d1.cy:.1f}), area={d1.area:.0f}")
        print("  抓取位画面已保存: logs/box_pick_calibration.jpg")

        cfg["box_pick_height_offset_deg"] = 0.0
        save_config({
            "box_look_joints": look,
            "box_pick_joints": pick,
            "box_look_pixel": cfg["box_look_pixel"],
            "box_pick_pixel": cfg["box_pick_pixel"],
            "box_pick_height_offset_deg": 0.0,
        })
        print("盆内抓出位姿已保存（box_pick_height_offset_deg 重置为 0）")

        if test:
            input("3/3 确认手已离开机械臂，按回车通电验证闭爪 ... ")
            robot.enable_torque()
            robot.move_joints(np.asarray(pick, float), float(cfg["gripper_open"]),
                              speed=8, label="验证盆内抓取位")
            pos = robot.set_gripper(float(cfg["gripper_closed"]), settle=0.8)
            load = robot.read_load()
            print(f"闭爪: pos={pos:.2f} load={load:.1f}（抬升后是否保持需实跑验证）")
            robot.set_gripper(float(cfg["gripper_open"]), settle=0.3)
    finally:
        cam.stop()
        robot.disconnect()


def cmd_run(cfg: dict, logger: logging.Logger, cycles: int, dry_run: bool,
            max_seconds: float, display: bool):
    if dry_run:
        cmd_preview(cfg, logger, seconds=max_seconds or 1e9, display=display)
        return
    sup = Supervisor(cfg, logger)
    sup.run(max_cycles=cycles, max_seconds=max_seconds)


def main():
    ap = argparse.ArgumentParser(description="square2box 24/7 抓取循环")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("self-test", help="只检查配置/相机/YOLO/机械臂，不动电机")
    p_prev = sub.add_parser("preview", help="相机+YOLO 预览（--hover/--box 会动电机）")
    p_prev.add_argument("--seconds", type=float, default=10.0)
    p_prev.add_argument("--display", action="store_true")
    p_prev.add_argument("--hover", action="store_true", help="先移到 hover_joints 再预览（会动电机）")
    p_prev.add_argument("--box", action="store_true", help="先移到 box_hover_joints 再预览（会动电机）")
    sub.add_parser("calibrate-extra", help="录 box_pick/table_drop/box_look/home 位姿")
    p_box = sub.add_parser("calibrate-box-pick", help="单独标定盆内观察位/抓取位（卸力）")
    p_box.add_argument("--test", action="store_true", help="记录后通电验证闭爪")
    sub.add_parser("calibrate-depth", help="探测深度雅可比（会动电机）")
    p_run = sub.add_parser("run", help="无限循环抓取-放入-取出-放回")
    p_run.add_argument("--cycles", type=int, default=0, help="0=无限")
    p_run.add_argument("--dry-run", action="store_true", help="只开相机/YOLO，不动电机")
    p_run.add_argument("--max-seconds", type=float, default=0.0)
    p_run.add_argument("--display", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    logger = setup_logging(cfg)
    if args.cmd == "self-test":
        cmd_self_test(cfg, logger)
    elif args.cmd == "preview":
        cmd_preview(cfg, logger, seconds=args.seconds, display=args.display,
                    hover=args.hover, box=args.box)
    elif args.cmd == "calibrate-extra":
        cmd_calibrate_extra(cfg, logger)
    elif args.cmd == "calibrate-box-pick":
        cmd_calibrate_box_pick(cfg, logger, test=args.test)
    elif args.cmd == "calibrate-depth":
        cmd_calibrate_depth(cfg, logger)
    elif args.cmd == "run":
        cmd_run(cfg, logger, cycles=args.cycles, dry_run=args.dry_run,
                max_seconds=args.max_seconds, display=args.display)


if __name__ == "__main__":
    main()
