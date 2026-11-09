#!/usr/bin/env python
"""armvision — 双腕相机 + 方块检测 + 抓取前视觉对准。

检测策略（兼顾稳定与 24/7）：
  1. YOLOE 开放词表分割（GPU，间隔推理，结果缓存）；
  2. 在 YOLO 框内用 HSV 精修中心/角度（绿色方块阈值可调）；
  3. YOLO 不可用或漏检时，整幅 HSV 兜底。

视觉对准（不做长时间伺服）：用标定好的“像素/度”小增益矩阵，
每次最多修正 2 步、每步 ≤ align_step_deg；误差进容差就立刻结束。
这样不会因为反复微调而出现“原地抖动/停顿”。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from armcore import HERE, JOINTS, clamp, find_videos_by_usb, fold_angle, in_roi


class CameraError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 相机线程
# ---------------------------------------------------------------------------

class CameraStream:
    """后台持续读帧，主线程永远拿最新帧，避免 read() 阻塞动作。

    devices 可以给多个候选（同一个 USB 口通常有两个 /dev/videoN，其中
    一个是元数据节点）。打不开或读不到帧就自动轮换到下一个，长时间
    无人值守时不会因为 udev 重编号或某个节点抽风而卡死。
    """

    def __init__(self, device: str, width=640, height=480, fps=30,
                 logger=None, name="cam", devices=None, usb_port=None):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.name = name
        self.log = logger
        self.usb_port = str(usb_port) if usb_port else None
        self.candidates = []
        for d in [device] + list(devices or []):
            if d and str(d) not in self.candidates:
                self.candidates.append(str(d))
        if not self.candidates:
            self.candidates = [str(device)]
        self._cand_i = 0
        self.device = self.candidates[0]
        self._frame = None
        self._ts = 0.0
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._reopen = False
        self._thread: Optional[threading.Thread] = None
        self._fps_est = 0.0
        self._times: list[float] = []
        self._fail = 0
        self._open_fail = 0

    # ---- 生命周期 ----
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"cam-{self.name}")
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def reconnect(self):
        """请求重开当前设备。不要重置候选下标——看门狗每秒调用一次，
        重置会把正在轮换的候选列表反复打回第一个，导致永远在同一个
        坏节点上打转。"""
        self._reopen = True

    def _refresh_candidates(self):
        """USB 相机热插拔/重新枚举后 /dev/videoN 可能变号，重新扫一遍。"""
        if not self.usb_port:
            return
        fresh = [d for d in find_videos_by_usb(self.usb_port) if d not in self.candidates]
        if fresh:
            self.candidates.extend(fresh)
            self._log("info", "发现新的相机节点: %s（候选=%s）", fresh, self.candidates)

    def _advance_candidate(self):
        if len(self.candidates) > 1:
            self._cand_i = (self._cand_i + 1) % len(self.candidates)
            self.device = self.candidates[self._cand_i]

    def _open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None
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
            if cap is None:
                cap = self._open()
                if cap is None:
                    self._open_fail += 1
                    if self._open_fail in (1, 5) or self._open_fail % 30 == 0:
                        self._log("error", "打不开 %s（候选 %s），重试",
                                  self.device, self.candidates)
                    self._advance_candidate()
                    if self._open_fail % 6 == 0:
                        self._refresh_candidates()
                        time.sleep(1.0)
                    else:
                        time.sleep(0.35)
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    # 节点能打开但读不到帧：多半是同口的元数据节点，
                    # 或者 USB 流卡死。释放后等一会儿再换，给内核恢复时间。
                    self._open_fail += 1
                    self._log("warning", "%s 无帧，换候选节点", self.device)
                    cap.release()
                    cap = None
                    self._advance_candidate()
                    if self._open_fail % 4 == 0:
                        self._refresh_candidates()
                        time.sleep(1.2)
                    else:
                        time.sleep(0.6)
                    continue
                self._open_fail = 0
                self._log("info", "相机已打开 %s (%dx%d@%d) 候选=%s",
                          self.device, self.width, self.height, self.fps,
                          self.candidates)
            ok, frame = cap.read()
            if not ok or frame is None:
                self._fail += 1
                self._log("warning", "读帧失败 (%d)，换候选重连", self._fail)
                cap.release()
                cap = None
                self._advance_candidate()
                time.sleep(0.4)
                continue
            self._fail = 0
            now = time.time()
            with self._cond:
                self._frame = frame
                self._ts = now
                self._seq += 1
                self._times.append(now)
                if len(self._times) > 40:
                    self._times = self._times[-40:]
                if len(self._times) >= 2:
                    dt = self._times[-1] - self._times[0]
                    self._fps_est = (len(self._times) - 1) / dt if dt > 0 else 0.0
                self._cond.notify_all()
        if cap is not None:
            cap.release()

    def _log(self, level, msg, *args):
        if self.log is not None:
            getattr(self.log, level)("[%s] " + msg, self.name, *args)

    # ---- 读 ----
    def _age_locked(self) -> float:
        return float("inf") if self._ts == 0 else time.time() - self._ts

    def latest(self) -> tuple[Optional[np.ndarray], float]:
        with self._cond:
            return (None if self._frame is None else self._frame.copy(),
                    self._age_locked())

    def read(self, timeout: float = 1.0, max_age: float = 0.8) -> np.ndarray:
        with self._cond:
            seq = self._seq
            self._cond.wait_for(lambda: self._seq > seq or self._stop.is_set(), timeout)
            if self._frame is None:
                raise CameraError(f"[{self.name}] 还没有画面")
            age = self._age_locked()
            if age > max_age:
                raise CameraError(f"[{self.name}] 画面已过期 {age:.2f}s")
            return self._frame.copy()

    def fresh(self, max_age: float = 0.6) -> Optional[np.ndarray]:
        frame, age = self.latest()
        if frame is None or age > max_age:
            return None
        return frame

    @property
    def age(self) -> float:
        with self._cond:
            return self._age_locked()

    @property
    def fps_est(self) -> float:
        with self._cond:
            return self._fps_est


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

    def __str__(self):
        return (f"{self.source} conf={self.conf:.2f} "
                f"c=({self.cx:.0f},{self.cy:.0f}) area={self.area:.0f} "
                f"ang={self.angle:.1f}")


class Detector:
    """YOLOE（可选）+ HSV 精修/兜底。一个实例可服务两条臂（同一模型）。"""

    def __init__(self, cfg: dict, logger=None):
        self.cfg = cfg
        self.log = logger
        dcfg = cfg["detector"]
        self.prompts = list(dcfg["yolo_prompts"])
        self.target_terms = [t.lower() for t in dcfg["yolo_target_terms"]]
        self.conf = float(dcfg["yolo_conf"])
        self.imgsz = int(dcfg["yolo_imgsz"])
        self.device = dcfg.get("yolo_device", "cuda:0")
        self.interval = float(dcfg["yolo_interval_s"])
        self.use_yolo = bool(dcfg.get("use_yolo", True))
        self.min_area = int(dcfg["min_area"])
        self.min_area_box = int(dcfg["min_area_box"])
        self.max_area_frac = float(dcfg["max_area_frac"])
        self.min_color_frac = float(dcfg["min_color_frac"])
        self.hsv_low1 = np.array(dcfg["hsv_low1"], dtype=np.uint8)
        self.hsv_high1 = np.array(dcfg["hsv_high1"], dtype=np.uint8)
        self.hsv_low2 = np.array(dcfg["hsv_low2"], dtype=np.uint8)
        self.hsv_high2 = np.array(dcfg["hsv_high2"], dtype=np.uint8)

        self.model = None
        self.names = {}
        self._cache: dict[str, tuple[float, Optional[Detection]]] = {}
        self._lock = threading.Lock()
        self.last_infer_s = 0.0
        if self.use_yolo:
            self._load_model()

    # ---- 模型 ----
    def _model_candidates(self):
        name = str(self.cfg["detector"].get("yolo_model") or "")
        return [Path(name), HERE / name, HERE.parent / name,
                Path("/home/zhoulin/hu/lerobot") / name,
                HERE / "weights" / name]

    def _load_model(self):
        path = next((p for p in self._model_candidates() if p and p.is_file()), None)
        if path is None:
            self._log("warning", "找不到 YOLO 模型，改用 HSV 检测")
            self.use_yolo = False
            return
        try:
            from ultralytics import YOLOE
            self._ensure_text_encoder(path)
            model = YOLOE(str(path))
            model.set_classes(self.prompts, model.get_text_pe(self.prompts))
            self.model = model
            self.names = model.names
            if bool(self.cfg["detector"].get("yolo_warmup", True)):
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                t0 = time.time()
                self._predict_raw(dummy)
                self._log("info", "YOLO 预热 %.2fs", time.time() - t0)
            self._log("info", "YOLO 已加载: %s prompts=%s", path.name, self.prompts)
        except Exception as exc:
            self.model = None
            self.use_yolo = False
            self._log("warning", "YOLO 加载失败，改用 HSV: %s", exc)

    def _ensure_text_encoder(self, model_path: Path) -> None:
        target = Path("mobileclip_blt.ts")
        if target.exists():
            return
        for cand in (HERE / "mobileclip_blt.ts", HERE.parent / "mobileclip_blt.ts",
                     model_path.parent / "mobileclip_blt.ts",
                     Path("/home/zhoulin/hu/lerobot/examples/mobileclip_blt.ts")):
            if cand.is_file():
                try:
                    target.symlink_to(cand.resolve())
                    self._log("info", "已链接文本编码器: %s -> %s", target, cand)
                except Exception:
                    pass
                return
        self._log("warning", "找不到 mobileclip_blt.ts，YOLOE 可能无法加载")

    def _log(self, level, msg, *args):
        if self.log is not None:
            getattr(self.log, level)(msg, *args)

    # ---- 推理 ----
    def _predict_raw(self, frame: np.ndarray):
        return self.model.predict(frame, conf=self.conf, device=self.device,
                                 imgsz=self.imgsz, verbose=False)[0]

    def _is_target(self, name: str) -> bool:
        s = str(name).lower()
        if "box" in s or "cup" in s:
            return False
        return any(t in s for t in self.target_terms)

    def _predict(self, frame: np.ndarray, roi=None) -> Optional[Detection]:
        if self.model is None:
            return None
        t0 = time.perf_counter()
        try:
            r = self._predict_raw(frame)
            self.last_infer_s = time.perf_counter() - t0
        except Exception as exc:
            self._log("warning", "YOLO 推理失败: %s", exc)
            return None
        if r.boxes is None or len(r.boxes) == 0:
            return None
        h, w = frame.shape[:2]
        best: Optional[Detection] = None
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls)
            if isinstance(self.names, dict):
                name = self.names.get(cls_id, str(cls_id))
            else:
                name = self.names[cls_id] if 0 <= cls_id < len(self.names) else str(cls_id)
            if not self._is_target(name):
                continue
            conf = float(box.conf)
            if conf < self.conf:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cxb, cyb = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if roi is not None and not in_roi((cxb, cyb), roi):
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
                cx, cy = cxb, cyb
                area = (x2 - x1) * (y2 - y1)
                angle = 0.0
            if area > h * w * self.max_area_frac:
                continue
            det = Detection(cx, cy, area, angle, (x1, y1, x2, y2), conf, "yolo",
                            time.time())
            if best is None or det.conf > best.conf:
                best = det
        return best

    # ---- HSV ----
    def _hsv_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m = cv2.bitwise_or(cv2.inRange(hsv, self.hsv_low1, self.hsv_high1),
                           cv2.inRange(hsv, self.hsv_low2, self.hsv_high2))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    def _color_frac(self, frame: np.ndarray, bbox) -> float:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        mask = self._hsv_mask(frame[y1:y2, x1:x2])
        return float(np.count_nonzero(mask)) / float((x2 - x1) * (y2 - y1))

    def _refine_hsv(self, frame, det: Detection, roi=None) -> Optional[Detection]:
        h, w = frame.shape[:2]
        pad = 14
        x1 = int(clamp(det.bbox[0] - pad, 0, w - 1))
        y1 = int(clamp(det.bbox[1] - pad, 0, h - 1))
        x2 = int(clamp(det.bbox[2] + pad, 0, w))
        y2 = int(clamp(det.bbox[3] + pad, 0, h))
        full = self._hsv_mask(frame)
        mask = np.zeros_like(full)
        mask[y1:y2, x1:x2] = full[y1:y2, x1:x2]
        if roi is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in roi]
            keep = np.zeros_like(mask)
            keep[max(0, ry1):min(h, ry2), max(0, rx1):min(w, rx2)] = 255
            mask = cv2.bitwise_and(mask, keep)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            if self._color_frac(frame, det.bbox) < self.min_color_frac:
                return None  # 框里没有目标颜色 -> 误检
            return det
        c = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        if area < max(self.min_area * 0.4, 90.0):
            return det
        m = cv2.moments(c)
        if m["m00"] <= 1e-6:
            return det
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        (_, _), (rw, rh), ang = cv2.minAreaRect(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        return Detection(cx, cy, area, fold_angle(ang, rw, rh),
                         (bx, by, bx + bw, by + bh), det.conf, "yolo+hsv", time.time())

    def _detect_hsv(self, frame, roi=None) -> Optional[Detection]:
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
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        (_, _), (rw, rh), ang = cv2.minAreaRect(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        return Detection(cx, cy, area, fold_angle(ang, rw, rh),
                         (bx, by, bx + bw, by + bh), 1.0, "hsv", time.time())

    # ---- 对外 ----
    def detect(self, frame: np.ndarray, roi=None, cache_key: Optional[str] = None,
               force_yolo: bool = False) -> Optional[Detection]:
        """带缓存的检测：YOLO 按间隔推理，中间帧用缓存 + HSV 精修。"""
        now = time.time()
        with self._lock:
            cached_ts, cached = self._cache.get(cache_key or "default", (0.0, None))
        use_cache = (not force_yolo and cached is not None
                     and now - cached_ts < max(self.interval * 3.0, 0.25))
        det = None
        if self.use_yolo and self.model is not None and (
                force_yolo or not use_cache or now - cached_ts >= self.interval):
            det = self._predict(frame, roi)
            with self._lock:
                self._cache[cache_key or "default"] = (now, det)
        elif use_cache:
            det = cached
            if roi is not None and det is not None and not in_roi(det.center, roi):
                det = None
        if det is not None:
            if roi is not None and not in_roi(det.center, roi):
                det = None
            else:
                refined = self._refine_hsv(frame, det, roi)
                if refined is not None:
                    return refined
                det = None
        return self._detect_hsv(frame, roi)

    def annotate(self, frame, det: Optional[Detection], extra: str = ""):
        if det is None:
            cv2.putText(frame, "NO TARGET", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
        else:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.drawMarker(frame, (int(det.cx), int(det.cy)), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(frame, f"{det.source} {det.conf:.2f} area={det.area:.0f} "
                               f"ang={det.angle:.0f}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if extra:
            cv2.putText(frame, extra, (12, frame.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        return frame

    def close(self):
        self.model = None
        self._cache.clear()


# ---------------------------------------------------------------------------
# 视觉对准
# ---------------------------------------------------------------------------

class VisionAligner:
    """用标定好的像素增益做少量步进对准，替代长时间视觉伺服。"""

    def __init__(self, cfg: dict, arm: str, detector: Detector, logger):
        self.cfg = cfg
        self.arm = arm
        self.det = detector
        self.log = logger
        g = cfg["arms"][arm].get("gains") or {}
        self.hover_gain = g.get("hover_px_per_deg")
        self.align_px = g.get("align_px")
        self.roll_slope = g.get("roll_deg_per_deg")
        gr = cfg["grasp"]
        self.max_iters = int(gr["align_max_iters"])
        self.tol = float(gr["align_tol_px"])
        self.step = float(gr["align_step_deg"])
        fallback = np.asarray(gr["align_gain_fallback"], dtype=float)
        self.gain = (np.asarray(self.hover_gain, dtype=float)
                     if self.hover_gain else fallback)

    def target_px(self) -> Optional[np.ndarray]:
        if self.align_px:
            return np.asarray(self.align_px, dtype=float)
        return None

    def align(self, arm, camera: CameraStream, target_px: np.ndarray,
              pose: np.ndarray, roi=None) -> tuple[np.ndarray, Optional[Detection]]:
        """在给定位姿附近微调，使方块落到 target_px。返回 (新位姿, 检测)。"""
        q = np.asarray(pose, dtype=float).copy()
        det = None
        for it in range(self.max_iters + 1):
            frame = camera.fresh(max_age=0.8)
            if frame is None:
                try:
                    frame = camera.read(timeout=0.8)
                except CameraError as exc:
                    self.log.warning("[%s] 对准时相机异常: %s", self.arm, exc)
                    break
            det = self.det.detect(frame, roi=roi)
            if det is None:
                break
            err = det.center - np.asarray(target_px, dtype=float)
            if float(np.abs(err).max()) <= self.tol:
                break
            if it >= self.max_iters:
                break
            try:
                pinv = np.linalg.pinv(self.gain)
            except Exception:
                break
            delta = pinv @ (-err)
            delta = np.clip(delta, -self.step, self.step)
            q[:2] += delta
            arm.start([q], gripper=arm.hold_gripper,
                      label=f"{self.arm} 视觉对准 {it + 1}")
            arm.wait(timeout=2.0)
        return q, det

    def align_roll(self, arm, camera: CameraStream, pose: np.ndarray,
                   det: Optional[Detection], roi=None) -> np.ndarray:
        """可选：转腕让夹爪对齐方块边。默认关闭，避免额外停顿。"""
        if not bool(self.cfg["grasp"].get("align_roll", False)):
            return np.asarray(pose, dtype=float).copy()
        if det is None or self.roll_slope is None or abs(self.roll_slope) < 0.02:
            return np.asarray(pose, dtype=float).copy()
        if abs(det.angle) < float(self.cfg["grasp"]["align_tol_px"]):
            return np.asarray(pose, dtype=float).copy()
        q = np.asarray(pose, dtype=float).copy()
        corr = clamp(-det.angle / self.roll_slope,
                     -float(self.cfg["grasp"]["align_roll_max_deg"]),
                     float(self.cfg["grasp"]["align_roll_max_deg"]))
        q[JOINTS.index("wrist_roll")] += corr
        arm.start([q], gripper=arm.hold_gripper, label=f"{self.arm} 转腕")
        arm.wait(timeout=2.0)
        return q


def box_roi_from_cfg(cfg: dict, arm: str) -> Optional[tuple]:
    roi = (cfg["arms"][arm].get("gains") or {}).get("box_roi")
    if not roi or len(roi) != 4:
        return None
    return tuple(int(v) for v in roi)


def save_snapshot(cfg: dict, detector: Detector, frame, det, name: str,
                  logger=None) -> Optional[Path]:
    try:
        snap = HERE / str(cfg["loop"].get("snapshot_dir", "snapshots"))
        snap.mkdir(parents=True, exist_ok=True)
        path = snap / f"{name}.jpg"
        cv2.imwrite(str(path), detector.annotate(frame.copy(), det))
        return path
    except Exception as exc:
        if logger:
            logger.debug("保存快照失败: %s", exc)
        return None
