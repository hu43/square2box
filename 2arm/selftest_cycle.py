#!/usr/bin/env python
"""selftest_cycle.py — 不需要硬件的离线验证。

两部分：
  1. 伺服数学：用一个假的“眼在手上”相机模型验证 hover 对准、分段跟踪下探、
     合爪视觉闸门、学习值持久化。相机模型是可解析的，能算出理论误差。
  2. 真实检测：用 calib/ 里已有的图跑 YOLO+HSV，确认几种关键视角能稳定
     检出绿色方块（右盒内、左盒内、右桌面）。

用法：
    python selftest_cycle.py            # 全部
    python selftest_cycle.py servo      # 只跑伺服数学
    python selftest_cycle.py vision     # 只跑图像检测
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import armcore as C
from armvision import Detection, Detector
from servo import PAN, ELBOW, PixelServo, ServoState


def make_logger() -> logging.Logger:
    log = logging.getLogger("selftest_cycle")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log


# ---------------------------------------------------------------------------
# 1. 假机器人 + 解析相机模型
# ---------------------------------------------------------------------------

IMG_C = np.array([320.0, 240.0])
DRIFT = np.array([0.0, -90.0])       # 相机随臂下降造成的像素漂移
PAN_SCALE = 0.5                      # 世界单位 / 关节单位
ELBOW_SCALE = 0.4
K = np.array([[4.0, 0.0], [0.0, 4.0]])   # 像素 / 世界单位


class FakeBot:
    """把 pan/elbow 映射成夹爪的世界坐标，再投影成像素。"""

    def __init__(self, cube_xy: np.ndarray, delta: np.ndarray,
                 disturb_at: float = -1.0, disturb_xy=(0.0, 0.0)):
        self.arm = "left"
        self.q = np.zeros(C.NJ)
        self.q0 = self.q.copy()
        self.cube = np.asarray(cube_xy, dtype=float).copy()
        self.delta = np.asarray(delta, dtype=float)
        self.disturb_at = disturb_at
        self.disturb_xy = np.asarray(disturb_xy, dtype=float)
        self._disturbed = False
        self.moves = 0
        self.log = make_logger()

    # --- 几何 ---
    @property
    def f(self) -> float:
        d = float(self.delta[1])
        if abs(d) < 1e-6:
            return 0.0
        return float(np.clip((self.q[1] - self.q0[1]) / d, 0.0, 1.0))

    @property
    def gripper_xy(self) -> np.ndarray:
        return np.array([self.q[PAN] * PAN_SCALE, self.q[ELBOW] * ELBOW_SCALE])

    def pixel(self) -> np.ndarray:
        return IMG_C + K @ (self.cube - self.gripper_xy) + DRIFT * self.f

    def det(self) -> Detection:
        p = self.pixel()
        return Detection(float(p[0]), float(p[1]), 5000.0, 0.0,
                         (p[0] - 30, p[1] - 30, p[0] + 30, p[1] + 30),
                         0.9, "fake", time.time())

    def reset(self):
        self.q = np.zeros(C.NJ)
        self.q0 = self.q.copy()
        self._disturbed = False

    # --- runner 接口 ---
    def move_q(self, arm, q, *, gripper=None, label="", speed=1.0,
               precision=False, tolerate=False, timeout=None, arrive_tol=None):
        self.q = np.asarray(q, dtype=float).copy()
        self.moves += 1
        if (self.disturb_at >= 0 and not self._disturbed
                and self.f >= self.disturb_at):
            self.cube = self.cube + self.disturb_xy
            self._disturbed = True
        return True

    def detect_stable(self, arm, zone, *, force=False, n=2, max_age=1.2):
        return self.det()

    def detect(self, arm, zone=None, *, force=False):
        return self.det()

    def pose(self, arm, key):
        return self.q.copy()

    def planned_q(self, arm, fallback_key="home"):
        return self.q.copy()


def test_servo(log) -> int:
    print("\n[1] 伺服数学")
    cfg = C.load_config()
    cfg["servo"]["stable_px"] = 5.0
    cfg["servo"]["reject_jump_px"] = 500.0   # 允许测试中的扰动
    # 测试用更严的容差，验证的是算法精度而不是默认参数
    cfg["servo"]["target_tol_px"] = 4.0
    cfg["servo"]["descent_tol_px"] = 5.0
    tmp = Path(tempfile.mkdtemp()) / "servo_state.json"
    state = ServoState(tmp)
    sv = PixelServo(cfg, "left", log, state)

    failures = 0
    delta = np.array([0.0, 8.0, -4.0, -1.0, 0.0])

    # --- 1a. hover 对准 ---
    bot = FakeBot(cube_xy=[4.0, -3.5], delta=delta)
    hover_px = IMG_C.copy()      # 夹爪在方块正上方时的像素
    J = sv.ensure_jacobian(bot, "table", bot.q.copy())
    print(f"   测得雅可比: {None if J is None else np.round(J, 2).tolist()} "
          f"(理论 [[-2,0],[0,-1.6]])")
    q_res, det, conv, err = sv.center(bot, "table", bot.q.copy(), hover_px, J)
    world_err = float(np.linalg.norm(bot.cube - bot.gripper_xy))
    print(f"   hover 对准: 收敛={conv} 像素残差={err:.1f}px 世界误差={world_err:.2f}")
    if not conv or world_err > 0.6:
        print("   ✗ hover 对准没有收敛到方块")
        failures += 1
    else:
        print("   ✓ hover 对准收敛")

    # --- 1b. 分段跟踪下探（中途扰动） ---
    bot.reset()
    q_res, det, conv, err = sv.center(bot, "table", bot.q.copy(), hover_px, J)
    learned_hover = det.center.copy()
    # 下探到底时，夹爪在方块上：像素 = IMG_C + DRIFT
    pick_px = IMG_C + DRIFT
    bot.disturb_at = 0.5
    bot.disturb_xy = (3.0, 2.5)
    q_fin, det_bot, q_plan, n_det = sv.descend(
        bot, "table", bot.q.copy(), delta, learned_hover, pick_px, J)
    world_err = float(np.linalg.norm(bot.cube - bot.gripper_xy))
    gate, why = sv.gate_ok(det_bot, pick_px)
    print(f"   跟踪下探: 世界误差={world_err:.2f} 闸门={gate} ({why}) 检测次数={n_det}")
    if world_err > 1.5 or not gate:
        print("   ✗ 跟踪下探没有跟住被扰动的方块")
        failures += 1
    else:
        print("   ✓ 跟踪下探跟住扰动")

    # --- 1c. 合爪闸门 ---
    now = time.time()
    good = Detection(float(pick_px[0]), float(pick_px[1]), 100, 0,
                     (0, 0, 1, 1), 1.0, "fake", now)
    cases = [
        (None, pick_px, False, "无检测"),
        (Detection(320, 240, 100, 0, (0, 0, 1, 1), 1.0, "fake", now - 1.0),
         pick_px, False, "检测过期"),
        (Detection(320, 240, 100, 0, (0, 0, 1, 1), 1.0, "fake", now),
         pick_px + 200.0, False, "偏离抓取像素"),
        (good, pick_px, True, "新鲜且对准"),
    ]
    for det_c, target, want, name in cases:
        got, why = sv.gate_ok(det_c, target, now=now)
        mark = "✓" if got == want else "✗"
        print(f"   {mark} 闸门[{name}] -> {got} ({why})")
        if got != want:
            failures += 1

    # --- 1d. 学习值持久化 ---
    sv.learn_success("table", learned_hover, det_bot.center)
    sv2 = PixelServo(cfg, "left", log, ServoState(tmp))
    hp, pp = sv2.targets("table")
    ok = (hp is not None and pp is not None
          and np.allclose(hp, learned_hover, atol=0.5))
    print(f"   持久化: hover_px={None if hp is None else np.round(hp, 1).tolist()} "
          f"pick_px={None if pp is None else np.round(pp, 1).tolist()} -> {'✓' if ok else '✗'}")
    if not ok:
        failures += 1

    print(f"[1] {'通过' if failures == 0 else f'{failures} 个失败'}")
    return failures


# ---------------------------------------------------------------------------
# 2. 真实图像检测
# ---------------------------------------------------------------------------

VISION_CASES = [
    ("calib/right_box_pick.jpg", (331, 303), 70),
    ("calib/right_box_hover.jpg", (331, 303), 70),
    ("calib/left_box_pick.jpg", (330, 380), 80),
    ("calib/right_table_pick.jpg", (366, 322), 80),
    ("calib/left_table_hover.jpg", (350, 287), 90),
]


def test_vision(log) -> int:
    import cv2
    print("\n[2] 真实图像检测（calib/*.jpg）")
    cfg = C.load_config()
    det = Detector(cfg, log)
    if not det.use_yolo:
        print("   ! YOLO 未加载，只用 HSV 兜底（结果仅供参考）")
    failures = 0
    for rel, want, tol in VISION_CASES:
        p = C.HERE / rel
        if not p.exists():
            print(f"   - 跳过（不存在）: {rel}")
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"   - 跳过（读不了）: {rel}")
            continue
        t0 = time.perf_counter()
        d = det.detect(img, roi=None, cache_key=rel, force_yolo=True)
        dt = (time.perf_counter() - t0) * 1000
        if d is None:
            print(f"   ✗ {rel}: 没检出（期望 {want}）")
            failures += 1
            continue
        err = float(np.linalg.norm(d.center - np.asarray(want, dtype=float)))
        ok = err <= tol
        print(f"   {'✓' if ok else '✗'} {rel}: {np.round(d.center, 0).tolist()} "
              f"期望 {want} 误差 {err:.0f}px ({dt:.0f}ms)")
        if not ok:
            failures += 1
    det.close()
    print(f"[2] {'通过' if failures == 0 else f'{failures} 个失败'}")
    return failures


def main() -> int:
    log = make_logger()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fails = 0
    if which in ("all", "servo"):
        fails += test_servo(log)
    if which in ("all", "vision"):
        fails += test_vision(log)
    print("\n" + "=" * 60)
    print("离线自测:", "全部通过" if fails == 0 else f"{fails} 项失败")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
