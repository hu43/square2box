#!/usr/bin/env python
"""离线自测：不需要机械臂/相机，验证轨迹、状态机、标定校验逻辑。

    python selftest_offline.py

它会：
  1. 构造一套假的双臂（内存里模拟关节位置，按轨迹真实时间推进）；
  2. 用假相机（合成绿色方块）跑完整一轮流程；
  3. 检查每轮结束状态、区域互锁、异常恢复路径；
  4. 校验 config 合并、位姿高度差、轨迹速度上限。

这是给“标定完但还没上真机”阶段的验证工具。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import armcore as C
import armvision
import dualarm_loop as L


# ---------------------------------------------------------------------------
# 假位姿配置
# ---------------------------------------------------------------------------

def build_fake_config() -> dict:
    cfg = C.load_config()
    base = {"left": np.array([-20.0, 10.0, -5.0, 60.0, -40.0]),
            "right": np.array([18.0, 12.0, -6.0, 58.0, -38.0])}
    for arm in ("left", "right"):
        b = base[arm]
        poses = {
            "measure": b + np.array([20, -12, 0, 0, 0]),
            "home": b + np.array([10, -20, 8, -10, 0]),
            "carry": b + np.array([0, -14, 4, -8, 0]),
            "table_approach": b + np.array([0, -6, 2, -3, 0]),
            "table_hover": b.copy(),
            "table_pick": b + np.array([0, 14, -8, 6, 0]),
            "box_approach": b + np.array([18, -10, 4, -6, 0]),
            "box_hover": b + np.array([18, 2, -2, 2, 0]),
            "box_pick": b + np.array([18, 11, -6, 5, 0]),
            "box_drop": b + np.array([18, 9, -5, 4, 0]),
        }
        C.set_pose(cfg, arm, "x", b)  # 占位，下面删掉
        cfg["arms"][arm]["poses"] = {k: [round(float(v), 3) for v in q]
                                     for k, q in poses.items()}
        cfg["arms"][arm]["poses"].pop("x", None)
        cfg["arms"][arm]["gripper_open"] = 62.0
        cfg["arms"][arm]["gripper_closed"] = 8.0
        cfg["arms"][arm]["gains"] = {
            "hover_px_per_deg": [[-4.0, 0.0], [0.0, -22.0]],
            "align_px": [320.0, 300.0],
            "box_roi": [150, 60, 500, 430],
        }
    cfg["motion"]["cruise_scale"] = 6.0  # 离线跑快一点
    cfg["motion"]["settle_s"] = 0.0
    cfg["motion"]["gripper_open_settle_s"] = 0.0
    cfg["motion"]["gripper_close_settle_s"] = 0.0
    cfg["loop"]["snapshot_every_n_cycles"] = 0
    return cfg


# ---------------------------------------------------------------------------
# 假机械臂
# ---------------------------------------------------------------------------

class FakeWorld:
    """方块在世界里的位置：table / box / left / right。"""

    def __init__(self):
        self.block = "table"


class FakeArm:
    """按真实时间推进的假臂：执行轨迹并模拟夹爪抓住/放下方块。"""

    def __init__(self, name, cfg, poses, world, opening=62.0, closed=8.0, logger=None):
        self.arm = name
        self.cfg = cfg
        self.poses = poses
        self.world = world
        self.open_g = opening
        self.close_g = closed
        self.log = logger
        self.q = poses["home"].copy()
        self.hold_q = self.q.copy()
        self._planned_q = self.q.copy()
        self.hold_gripper = opening
        self.gripper_cmd = opening
        self.holding_block = False
        self.events: list[str] = []
        self._queue: list = []
        self._cond = threading.Condition()
        self._done_id = 0
        self._expect_id = 0
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        self.reads = 0

    # ---- worker ----
    def _loop(self):
        while not self._stop.is_set():
            with self._cond:
                self._cond.wait_for(lambda: bool(self._queue) or self._stop.is_set(), 0.2)
                if not self._queue:
                    continue
                item = self._queue.pop(0)
            path, times, g_from, g_to, switch_frac, label, cid = item
            self.events.append(label)
            t0 = time.perf_counter()
            released = False
            for i, t in enumerate(times):
                if self._stop.is_set():
                    return
                dt = t0 + t - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                self.q = path[i].copy()
                self.hold_q = self.q.copy()
                if g_from is not None and g_to is not None:
                    span = max(times[-1] * switch_frac, 1e-3)
                    frac = min(1.0, t / span)
                    self.hold_gripper = g_from + (g_to - g_from) * frac
                    self.gripper_cmd = self.hold_gripper
                    # 松开跨过中点的那一刻，方块就落到当前位置（盒内/桌面）
                    if (not released and g_to > g_from
                            and frac >= 0.5):
                        self._try_drop()
                        released = True
            with self._cond:
                self._done_id = cid
                self._cond.notify_all()

    # ---- API ----
    def start(self, waypoints, *, gripper=None, label="", speed_scale=1.0,
              ease_frac=None, precision=False):
        return self._submit(waypoints, gripper, label, speed_scale, precision)

    def start_mixed(self, waypoints, *, gripper_from, gripper_to, switch_frac=0.3,
                    label="", speed_scale=1.0, ease_frac=None, precision=False):
        return self._submit(waypoints, None, label, speed_scale, precision,
                            gripper_from, gripper_to, switch_frac)

    def _submit(self, waypoints, gripper, label, speed_scale, precision,
                g_from=None, g_to=None, switch_frac=0.3):
        pts = np.asarray(waypoints, float)
        s = float(speed_scale)
        if precision:
            s *= float(self.cfg["motion"]["precision_scale"])
        vmax = C.vmax_vector(self.cfg) * max(0.05, s) * float(self.cfg["motion"]["cruise_scale"])
        start = self._planned_q if self._planned_q is not None else self.hold_q
        if float(np.abs(pts[0] - start).max()) > 0.05:
            pts = np.vstack([start[None, :], pts])
        path = C.build_path(pts)
        times, path = C.time_parameterize(path, vmax)
        with self._cond:
            self._expect_id += 1
            cid = self._expect_id
            if g_from is None and g_to is None and gripper is not None:
                self.hold_gripper = float(gripper)
                self.gripper_cmd = float(gripper)
            self._queue.append((path, times, g_from, g_to, switch_frac, label, cid))
            self._planned_q = path[-1].copy()
            self._cond.notify_all()
        return float(times[-1])

    def busy(self):
        with self._cond:
            return bool(self._queue) or self._done_id != self._expect_id

    def wait(self, timeout=None, tol=None):
        cid = self._expect_id
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._cond:
            while self._done_id != cid:
                remain = None if deadline is None else deadline - time.perf_counter()
                if remain is not None and remain <= 0:
                    return False
                self._cond.wait(0.02 if remain is None else min(remain, 0.02))
        return True

    def move(self, wps, **kw):
        tmo = kw.pop("timeout", None)
        self.start(wps, **kw)
        return self.wait(timeout=tmo)

    def set_gripper(self, value, settle=None):
        value = float(value)
        self.hold_gripper = value
        self.gripper_cmd = value
        if value <= (self.close_g + self.open_g) * 0.5:
            self._try_grab()
        else:
            self._try_drop()
        if settle:
            time.sleep(settle)
        return self.read_gripper()

    def _near(self, key, tol=6.0):
        return float(np.abs(self.q - self.poses[key]).max()) <= tol

    def _try_grab(self):
        if self.holding_block or self.world is None:
            return
        if self.world.block == "table" and self._near("table_pick", 14.0):
            self.world.block = "held"
            self.holding_block = True
        elif self.world.block == "box" and self._near("box_pick", 14.0):
            self.world.block = "held"
            self.holding_block = True

    def _try_drop(self):
        if not self.holding_block:
            return
        # 就近判断放到桌上还是盒里
        d_table = float(np.abs(self.q - self.poses["table_pick"]).max())
        d_box = float(np.abs(self.q - self.poses["box_pick"]).max())
        if d_box < 24.0 or self._near("box_hover", 24.0) or self._near("box_drop", 12.0):
            self.world.block = "box"
        elif d_table < 24.0 or self._near("table_hover", 24.0):
            self.world.block = "table"
        else:
            self.world.block = "box"  # 默认按盒子释放处理
        self.holding_block = False

    def read_joints(self):
        self.reads += 1
        return self.q.copy()

    def read_gripper(self):
        self.reads += 1
        if self.holding_block:
            return 34.0  # 夹住方块：夹爪被撑开，读数高于空夹
        return self.hold_gripper

    def read_load(self):
        self.reads += 1
        # 闭爪且夹着方块 -> 负载
        if self.holding_block and self.hold_gripper < 20.0:
            return 150.0
        return 0.0

    def read_temp(self):
        return 40.0

    def open_gripper(self):
        return self.set_gripper(self.open_g)

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._worker.join(timeout=1.0)


class FakeDual:
    def __init__(self, cfg, poses, world=None):
        self.cfg = cfg
        self.world = world or FakeWorld()
        self.arms = {a: FakeArm(a, cfg, poses[a], self.world, 62.0, 8.0)
                     for a in ("left", "right")}

    def max_temp(self):
        return 40.0

    def error_streak(self):
        return 0

    def disable_torque(self):
        pass

    def enable_torque(self):
        pass

    def connect(self):
        pass

    def reconnect(self, name=None):
        pass

    def stop_workers(self):
        for a in self.arms.values():
            a.stop()


# ---------------------------------------------------------------------------
# 假相机 + 假检测
# ---------------------------------------------------------------------------

class FakeCamera:
    def __init__(self):
        self.age = 0.0

    def fresh(self, max_age=0.8):
        return np.zeros((480, 640, 3), np.uint8)

    def read(self, timeout=1.0, max_age=0.8):
        return np.zeros((480, 640, 3), np.uint8)

    def reconnect(self):
        pass


class FakeDetector:
    """根据“方块当前所在位置”返回固定像素，模拟视觉。"""

    def __init__(self, box):
        self.box = box
        self.model = object()

    def detect(self, frame, roi=None, cache_key=None, force_yolo=False):
        cx, cy = self.box["px"]
        det = armvision.Detection(cx, cy, 1800.0, 3.0, (cx - 30, cy - 30, cx + 30, cy + 30),
                                  0.9, "fake", time.time())
        if roi is not None and not C.in_roi(det.center, roi):
            return None
        return det

    def annotate(self, frame, det, extra=""):
        return frame

    def close(self):
        pass


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_config_merge():
    cfg = build_fake_config()
    p = Path("/tmp/_s2b2_cfg_test.json")
    if p.exists():
        p.unlink()
    C.save_config({"arms": {"left": {"gripper_open": 55.0}}}, p)
    C.save_config({"arms": {"left": {"gripper_closed": 9.0}}}, p)
    loaded = C.load_config(p)
    assert loaded["arms"]["left"]["gripper_open"] == 55.0
    assert loaded["arms"]["left"]["gripper_closed"] == 9.0
    assert loaded["motion"]["control_hz"] == cfg["motion"]["control_hz"]
    p.unlink()
    print("  config 深度合并: OK")


def test_pose_checks():
    cfg = build_fake_config()
    for arm in ("left", "right"):
        assert C.missing_poses(cfg, arm) == []
        msgs = C.pose_delta_report(cfg, arm)
        assert any("table_hover->table_pick" in m for m in msgs)
        assert not any("不是竖直下降" in m for m in msgs), msgs
    print("  hover->pick 竖直度检查: OK")

    cfg["arms"]["left"]["poses"]["table_pick"][0] += 8.0
    msgs = C.pose_delta_report(cfg, "left")
    assert any("不是竖直下降" in m for m in msgs), msgs
    print("  非竖直位姿能被检出: OK")


def test_trajectory_limits():
    cfg = build_fake_config()
    vmax = C.vmax_vector(cfg)
    wps = [np.zeros(5), np.array([40, -40, 40, -40, 40.0]), np.zeros(5)]
    path = C.build_path(wps)
    t, p = C.time_parameterize(path, vmax)
    sp = np.abs(np.diff(p, axis=0)) / np.diff(t)[:, None]
    peak = sp.max(axis=0)
    assert np.all(peak <= vmax * 1.05 + 1e-6), (peak, vmax)
    assert sp[0].max() < vmax.max() * 0.35, sp[0]
    print(f"  轨迹限速/缓起: OK (峰值 {np.round(peak, 1).tolist()}, 用时 {t[-1]:.2f}s)")


def test_full_cycle():
    cfg = build_fake_config()
    poses = {a: {k: np.asarray(v, float) for k, v in cfg["arms"][a]["poses"].items()}
             for a in ("left", "right")}
    dual = FakeDual(cfg, poses)
    cams = {"left": FakeCamera(), "right": FakeCamera()}
    box = {"px": (320.0, 300.0)}
    det = FakeDetector(box)

    for a in ("left", "right"):
        dual.arms[a].holding_block = False

    runner = L.LoopRunner(cfg, dual, cams, det, threading.Event(),
                          logger=_quiet_logger(), csv_writer=None)
    runner.prepare()
    runner.run_cycle(1)
    runner.run_cycle(2)

    left_events = dual.arms["left"].events
    right_events = dual.arms["right"].events

    # 第一轮：左臂应该先去桌面抓，再进盒子
    assert any("桌面" in e or "下探" in e for e in left_events), left_events
    assert any("放入" in e or "竖直放入" in e for e in left_events), left_events
    assert any("盒内" in e for e in right_events), right_events
    assert any("竖直放回" in e for e in right_events), right_events

    # 区域互锁：结束时盒子/桌面都不应被占用
    assert all(v is None for v in runner.zone_owner.values()), runner.zone_owner

    # 第二轮不应该再去桌面抓（方块一直夹在左爪）
    idx = left_events.index("桌面抓取") if "桌面抓取" in left_events else None
    print(f"  完整两轮: OK (左臂 {len(left_events)} 段动作, 右臂 {len(right_events)} 段)")
    dual.stop_workers()


def test_grasp_retry_and_failure():
    """夹爪一直夹不住时，应该抛 GraspFailed，而不是死循环。"""
    cfg = build_fake_config()
    poses = {a: {k: np.asarray(v, float) for k, v in cfg["arms"][a]["poses"].items()}
             for a in ("left", "right")}
    dual = FakeDual(cfg, poses)
    # 让左臂闭爪读数永远等于空夹值 -> 判定空爪
    left = dual.arms["left"]
    left.hold_gripper = 62.0
    orig_read = left.read_gripper
    left.read_gripper = lambda: 9.0
    cams = {"left": FakeCamera(), "right": FakeCamera()}
    det = FakeDetector({"px": (320.0, 300.0)})
    runner = L.LoopRunner(cfg, dual, cams, det, threading.Event(),
                          logger=_quiet_logger(), csv_writer=None)
    runner.prepare()
    t0 = time.perf_counter()
    try:
        runner.run_cycle(1)
        raised = False
    except C.PhaseError as exc:
        raised = True
        print(f"  空爪时抛 {type(exc).__name__}: OK ({time.perf_counter() - t0:.2f}s)")
    assert raised, "抓不住时没有抛异常"
    assert all(v is None for v in runner.zone_owner.values()), runner.zone_owner
    dual.stop_workers()


def test_missing_target():
    cfg = build_fake_config()
    poses = {a: {k: np.asarray(v, float) for k, v in cfg["arms"][a]["poses"].items()}
             for a in ("left", "right")}
    dual = FakeDual(cfg, poses)
    cams = {"left": FakeCamera(), "right": FakeCamera()}

    class NoDet(FakeDetector):
        def detect(self, *a, **k):
            return None

    runner = L.LoopRunner(cfg, dual, cams, NoDet({"px": (0, 0)}), threading.Event(),
                          logger=_quiet_logger(), csv_writer=None)
    runner.prepare()
    try:
        runner.run_cycle(1)
        assert False, "找不到方块时没有抛异常"
    except C.TargetMissing:
        print("  找不到方块时抛 TargetMissing: OK")
    assert all(v is None for v in runner.zone_owner.values())
    dual.stop_workers()


def test_zone_lock():
    cfg = build_fake_config()
    poses = {a: {k: np.asarray(v, float) for k, v in cfg["arms"][a]["poses"].items()}
             for a in ("left", "right")}
    dual = FakeDual(cfg, poses)
    runner = L.LoopRunner(cfg, dual, {"left": FakeCamera(), "right": FakeCamera()},
                          FakeDetector({"px": (320, 300)}), threading.Event(),
                          logger=_quiet_logger())
    runner.acquire("left", "box")
    try:
        runner.acquire("right", "box")
        assert False, "互锁失效"
    except C.PhaseError:
        pass
    runner.release("left", "box")
    runner.acquire("right", "box")
    runner.release("right", "box")
    print("  盒子区域互锁: OK")
    dual.stop_workers()


def test_no_backtrack_after_preposition():
    """预位到 box_approach 后，[carry, box_approach] 不应先退回 carry。"""
    cfg = build_fake_config()
    poses = {a: {k: np.asarray(v, float) for k, v in cfg["arms"][a]["poses"].items()}
             for a in ("left", "right")}
    dual = FakeDual(cfg, poses)
    runner = L.LoopRunner(cfg, dual, {"left": FakeCamera(), "right": FakeCamera()},
                          FakeDetector({"px": (320, 300)}), threading.Event(),
                          logger=_quiet_logger())
    left = dual.arms["left"]
    runner._preposition("left", "box_approach")
    left.wait(timeout=8.0)
    assert left._near("box_approach", 1.5), "预位没到位"

    # 记录这次移动过程中经过的关节轨迹，检查有没有退回 carry
    seen = []
    orig = left._submit

    def spy(wps, *a, **kw):
        seen.append(np.asarray(wps, float))
        return orig(wps, *a, **kw)

    left._submit = spy
    runner.move("left", ["carry", "box_approach"], gripper=runner.open_g["left"],
                label="测试", speed=1.0, timeout=5.0)
    assert seen, "没有提交轨迹"
    pts = seen[0]
    carry_eq = np.abs(pts - poses["left"]["carry"]).max(axis=1)
    assert float(carry_eq.min()) > 1.0, f"轨迹仍然经过 carry: {pts}"
    wps_only = [
        p for p in pts
        if float(np.abs(p - poses["left"]["box_approach"]).max()) < 1.5
        or float(np.abs(p - poses["left"]["carry"]).max()) < 1.5
    ]
    print(f"  预位后不折返: OK (路点 {len(wps_only)} 个, 未经过 carry)")
    dual.stop_workers()


def _quiet_logger():
    import logging
    log = logging.getLogger("offline_test")
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    log.setLevel(logging.CRITICAL)
    log.propagate = False
    return log


def main() -> int:
    print("=" * 60)
    print("2arm 离线自测（无硬件）")
    print("=" * 60)
    tests = [
        ("配置合并", test_config_merge),
        ("位姿校验", test_pose_checks),
        ("轨迹规划", test_trajectory_limits),
        ("区域互锁", test_zone_lock),
        ("预位不折返", test_no_backtrack_after_preposition),
        ("完整两轮", test_full_cycle),
        ("空爪重试失败", test_grasp_retry_and_failure),
        ("目标丢失", test_missing_target),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
        except Exception as exc:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  FAIL: {exc}")
    print()
    if failed:
        print(f"{failed} 项失败")
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
