#!/usr/bin/env python
"""calibrate_auto.py — 不靠手摆，通电自动测点位。

思路：腕相机能看到方块，就用像素增益把方块推到目标像素；看不到就用
夹爪本身当深度传感器——逐步下探，每一步合一次爪，第一个夹住的姿态
就是抓取位。

用法：
  python calibrate_auto.py --arm left  --phase table
  python calibrate_auto.py --arm left  --phase gain
  python calibrate_auto.py --arm left  --phase servo

安全：
  - 全程低速（speed_scale 0.25），每步都读位姿。
  - 任何异常 / Ctrl+C 都会先卸力再断开。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import armcore as C
from armdriver import DualArm
from armvision import CameraStream, Detector


def cam_device(cfg: dict, arm: str) -> str:
    d = cfg["devices"]
    usb = d.get(f"{arm}_camera_usb")
    if usb:
        got = C.find_video_by_usb(str(usb))
        if got:
            return got
    for c in [str(d.get(f"{arm}_camera") or "")] + list(d.get(f"{arm}_camera_fallbacks") or []):
        if c and Path(c).exists():
            return c
    return str(d.get(f"{arm}_camera"))


class AutoCal:
    def __init__(self, cfg: dict, logger, arms: list[str]):
        self.cfg = cfg
        self.log = logger
        self.stop = threading.Event()
        self.dual = DualArm(cfg, logger, self.stop)
        self.cams: dict[str, CameraStream] = {}
        self.det = Detector(cfg, logger)
        self.det.min_area = 40          # 远处方块很小，临时放宽
        self.arms = arms

    # ---- 生命周期 ----
    def __enter__(self):
        for a in self.arms:
            self.dual.arms[a].bus.connect()
            self.dual.arms[a].bus.enable_torque()
        for a in self.arms:
            cam = CameraStream(cam_device(self.cfg, a), logger=self.log, name=a)
            cam.start()
            self.cams[a] = cam
        time.sleep(1.0)
        return self

    def __exit__(self, *exc):
        for cam in self.cams.values():
            try:
                cam.stop()
            except Exception:
                pass
        for a in self.arms:
            try:
                self.dual.arms[a].bus.disable_torque()
            except Exception:
                pass
            try:
                self.dual.arms[a].bus.disconnect()
            except Exception:
                pass
        return False

    # ---- 基本操作 ----
    def q(self, arm: str) -> np.ndarray:
        return self.dual.arms[arm].bus.read_joints()

    def grip(self, arm: str) -> float:
        return self.dual.arms[arm].read_gripper()

    def open_g(self, arm: str) -> float:
        return float(self.cfg["arms"][arm]["gripper_open"])

    def close_g(self, arm: str) -> float:
        return float(self.cfg["arms"][arm]["gripper_closed"])

    def goto(self, arm: str, q, speed: float = 0.25, label: str = "") -> np.ndarray:
        a = self.dual.arms[arm]
        target = np.clip(np.asarray(q, dtype=float), -100.0, 100.0)
        a.start([target], gripper=self.grip(arm), label=label, speed_scale=speed)
        ok = a.wait(timeout=10.0)
        time.sleep(0.25)
        return self.q(arm)

    def frame(self, arm: str):
        cam = self.cams[arm]
        f = cam.fresh(max_age=1.0)
        if f is None:
            f = cam.read(timeout=1.5)
        return f

    def detect(self, arm: str):
        f = self.frame(arm)
        if f is None:
            return None, None
        return f, self.det.detect(f, force_yolo=True)

    def save(self, arm: str, tag: str, det=None):
        f = self.frame(arm)
        if f is None:
            return
        img = self.det.annotate(f.copy(), det) if det is not None else f.copy()
        cv2.imwrite(f"/tmp/auto_{arm}_{tag}.jpg", img)

    # ---- 夹爪基线 ----
    def air_baseline(self, arm: str) -> tuple[float, float]:
        """在当前位置空合一次，得到空夹的位置和负载。"""
        a = self.dual.arms[arm]
        a.set_gripper(self.open_g(arm), settle=0.4)
        pos = a.set_gripper(self.close_g(arm), settle=0.75)
        load = a.read_load()
        a.set_gripper(self.open_g(arm), settle=0.4)
        self.log.info("[%s] 空夹基线 pos=%.2f load=%.1f", arm, pos, load)
        print(f"  [{arm}] 空夹基线 pos={pos:.2f} load={load:.1f}")
        return float(pos), float(load)

    def try_grasp(self, arm: str, air_pos: float, air_load: float) -> tuple[bool, float, float]:
        a = self.dual.arms[arm]
        pos = a.set_gripper(self.close_g(arm), settle=0.62)
        load = a.read_load()
        grabbed = (pos > air_pos + 3.0) or (load > air_load + 30.0)
        a.set_gripper(self.open_g(arm), settle=0.32)
        return grabbed, float(pos), float(load)

    # ---- 像素增益 ----
    def measure_gain(self, arm: str, base: np.ndarray, delta: float = 5.0,
                     idx=(0, 1, 2, 3)) -> np.ndarray:
        """测 pan/lift 各 +delta 时方块像素的位移，返回 2x2（px / 单位）。"""
        _, d0 = self.detect(arm)
        if d0 is None:
            raise RuntimeError(f"[{arm}] 基准位看不到方块，无法测增益")
        print(f"  [{arm}] 基准检测: {d0}")
        c0 = d0.center
        J = np.zeros((2, len(idx)))
        for col, j in enumerate(idx):
            p = base.copy()
            p[j] += delta
            self.goto(arm, p, label=f"测增益 {C.JOINTS[j]}+{delta}")
            _, d1 = self.detect(arm)
            self.goto(arm, base, label="回基准")
            if d1 is None:
                print(f"    {C.JOINTS[j]}: 移动后丢失目标")
                continue
            J[:, col] = (d1.center - c0) / delta
            print(f"    {C.JOINTS[j]}: +{delta:.0f} -> 像素位移 "
                  f"({J[0, col]:+.2f}, {J[1, col]:+.2f}) px/单位")
        return J

    # ---- 视觉伺服 ----
    def servo(self, arm: str, base: np.ndarray, J: np.ndarray, target,
              iters: int = 10, step: float = 4.0, tol: float = 8.0) -> tuple[np.ndarray, object]:
        q = np.asarray(base, dtype=float).copy()
        try:
            pinv = np.linalg.pinv(J)
        except Exception as exc:
            raise RuntimeError(f"增益矩阵不可逆: {exc}")
        det = None
        for it in range(iters + 1):
            _, det = self.detect(arm)
            if det is None:
                print(f"    第 {it} 次: 看不到方块")
                break
            err = det.center - np.asarray(target, dtype=float)
            print(f"    第 {it} 次: 方块 {det.center.round(0).tolist()} "
                  f"误差 {err.round(0).tolist()}")
            if float(np.abs(err).max()) <= tol:
                break
            if it >= iters:
                break
            delta = pinv @ (-err)
            delta = np.clip(delta, -step, step)
            q[:len(delta)] += delta
            q = self.goto(arm, q, label=f"对准 {it + 1}")
        return q, det

    # ---- 下探 ----
    def descend_until_grasp(self, arm: str, start: np.ndarray, direction: np.ndarray,
                            air_pos: float, air_load: float, step: float = 0.25,
                            max_steps: int = 12) -> np.ndarray | None:
        """沿 direction 分步下探，每步合一次爪；抓住就返回那个位姿。"""
        d = np.asarray(direction, dtype=float)
        peak = float(np.abs(d).max())
        unit = d / peak if peak > 1e-6 else d
        q = np.asarray(start, dtype=float).copy()
        for k in range(1, max_steps + 1):
            q = np.clip(q + unit * (step * peak), -100.0, 100.0)
            self.goto(arm, q, speed=0.2, label=f"下探 {k}")
            ok, pos, load = self.try_grasp(arm, air_pos, air_load)
            print(f"    第 {k} 步 {'抓住' if ok else '空夹'}: pos={pos:.2f} load={load:.1f} "
                  f"q={np.round(q, 1).tolist()}")
            if ok:
                return q
        return None


    def null_direction(self, J: np.ndarray) -> np.ndarray:
        """求“让方块像素基本不动”的关节方向（雅可比的零空间）。

        腕相机看不出深度：沿光轴平移时方块在画面里几乎不动。零空间正好
        就是这类运动，沿它走可以在保持对准的同时改变夹爪高度。
        """
        idx = list(range(J.shape[1]))
        n = len(idx)
        # 求 J 的零空间：对 J·Jᵀ 做特征分解，取小特征值对应的方向
        u, s, vt = np.linalg.svd(J)
        null = vt[len(s):]          # 2 行约束之外的基
        if null.size == 0:
            return np.zeros(n)
        # 取第一个零空间方向，归一到最大分量为 1
        d = null[0]
        peak = float(np.abs(d).max())
        return d / peak if peak > 1e-9 else d

    def descend_along(self, arm: str, start: np.ndarray, direction: np.ndarray,
                      air_pos: float, air_load: float, sign: float = 1.0,
                      step: float = 3.0, max_steps: int = 14) -> np.ndarray | None:
        q = np.asarray(start, dtype=float).copy()
        d = np.zeros(q.shape[0], dtype=float)
        d[:len(direction)] = np.asarray(direction, dtype=float)
        for k in range(1, max_steps + 1):
            q = np.clip(q + d * (sign * step), -100.0, 100.0)
            self.goto(arm, q, speed=0.18, label=f"下潜 {k}")
            ok, pos, load = self.try_grasp(arm, air_pos, air_load)
            _, det = self.detect(arm)
            area = -1 if det is None else det.area
            print(f"    第 {k} 步 {'夹住' if ok else '空夹'} pos={pos:.2f} "
                  f"load={load:.0f} 面积={area:.0f} q={np.round(q,1).tolist()}")
            if ok:
                return q
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="自动标定点位")
    ap.add_argument("--arm", choices=["left", "right"], default="left")
    ap.add_argument("--phase", default="gain",
                    choices=["gain", "servo", "descend", "table"])
    ap.add_argument("--target", default="", help="目标像素 x,y（默认画面几何中心偏下）")
    ap.add_argument("--start", default="", help="从指定点位出发，例如 measure")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="auto")

    arm = args.arm
    with AutoCal(cfg, logger, [arm]) as ac:
        if args.start:
            p = C.get_pose(cfg, arm, args.start)
            print(f"[{arm}] 先移动到 {args.start}: {np.round(p, 1).tolist()}")
            ac.goto(arm, p, speed=0.25, label=f"去 {args.start}")
        base = ac.q(arm)
        print(f"[{arm}] 起始位姿 {np.round(base, 1).tolist()} 夹爪 {ac.grip(arm):.1f}")
        print(f"[{arm}] 相机 {cam_device(cfg, arm)}")
        air_pos, air_load = ac.air_baseline(arm)

        if args.phase == "gain":
            J = ac.measure_gain(arm, base)
            print("增益矩阵:\n", np.round(J, 3))
            if J.size and float(np.abs(J).max()) > 0.1:
                np.save(f"/tmp/gain_{arm}.npy", J)
                print(f"已保存 /tmp/gain_{arm}.npy")
        elif args.phase == "servo":
            J = np.load(f"/tmp/gain_{arm}.npy")
            tx, ty = (320.0, 240.0) if not args.target else map(float, args.target.split(","))
            q, det = ac.servo(arm, base, J, (tx, ty))
            print(f"[{arm}] 对准结果 {np.round(q, 1).tolist()} det={det}")
            np.save(f"/tmp/servo_{arm}.npy", np.asarray(q, dtype=float))
            ac.save(arm, "servo", det)
        elif args.phase == "descend":
            p = Path(f"/tmp/servo_{arm}.npy")
            if p.exists():
                base = np.load(p)
                print(f"[{arm}] 从对准结果出发 {np.round(base, 1).tolist()}")
                ac.goto(arm, base, speed=0.25, label="去对准位")
            J = np.load(f"/tmp/gain_{arm}.npy")
            d = ac.null_direction(J)
            print(f"[{arm}] 零空间方向 {np.round(d, 3).tolist()}")
            found = ac.descend_along(arm, base, d, air_pos, air_load, sign=+1.0)
            if found is None:
                ac.goto(arm, base, speed=0.2, label="回起点")
                found = ac.descend_along(arm, base, d, air_pos, air_load, sign=-1.0)
            print(f"[{arm}] 抓取位: " +
                  ("无" if found is None else str(np.round(found, 2).tolist())))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断，已卸力。")
        sys.exit(130)
