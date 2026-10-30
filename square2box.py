#!/usr/bin/env python
"""square2box: 左臂自动抓方块放进盒子（左腕相机 eye-in-hand 视觉伺服，不用头部相机）。

用法（conda env: hupy3.12）:
  python square2box.py preview     # 只开相机，调 HSV / 看检测效果，按 q 退出
  python square2box.py calibrate   # 示教关键点 + 探测伺服增益矩阵，写入 config.json
  python square2box.py run         # 执行抓取-放置
  python square2box.py run --dry-run  # 只看视觉检测，不动电机

安全设计:
  - 全程小步慢速插值（默认每步 <=1 度）
  - 伺服增量钳制 + 迭代次数上限 + 总超时
  - 任何异常/Ctrl+C: finally 里力矩下电、机械臂失力悬停（不会乱甩）
  - 内存占用极小（无任何 ML 模型）
"""

import argparse
import builtins
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"

try:
    from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import ColorMode, Cv2Rotation, Cv2Backends
except ImportError:
    sys.path.insert(0, "/home/zhoulin/hu/lerobot/src")
    from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import ColorMode, Cv2Rotation, Cv2Backends

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINT_KEYS = [f"left_arm_{j}.pos" for j in JOINTS]
GRIPPER_KEY = "left_arm_gripper.pos"
CAM_KEY = "right_wrist"  # XLerobot.get_camera_observation 用裸相机名做键

DEFAULT_CONFIG = {
    "hsv_low1": [0, 80, 60],
    "hsv_high1": [10, 255, 255],
    "hsv_low2": [170, 80, 60],
    "hsv_high2": [180, 255, 255],
    "min_area": 400,
    "pixel_tol": 8,
    "max_servo_iters": 40,
    "servo_step_deg": 1.2,
    "probe_delta_deg": 3.0,
    "move_step_deg": 1.0,
    "move_dt": 0.02,
    "gripper_settle_s": 0.8,
    "total_timeout_s": 120.0,
    "max_cycles": 0,
    "servo_joints": ["shoulder_pan", "shoulder_lift"],
    "grasp_retries": 3,
    "deepen_per_retry_deg": 1.0,
    "table_band_y": 330,
    # 2026-09-17 实测: 指隙在画面 (425,410) 附近且不随 pan/lift/elbow 变化
    # (相机对手指刚性), 旧值 [320,410] 会把方块对到左指上
    "grasp_depth_pixel": [420.0, 410.0],
    # 指尖闭合点在影像上高于真正夹持点(桌面接触处): 方块对到指尖中点会
    # 落在夹爪后方压桌。实测 roll≈0 时标定成功 y=410 vs 指尖 y≈322, 差 ~88px
    "grasp_dy_px": 80.0,
    # calibrate 后填充:
    "gripper_open": None,
    "gripper_closed": None,
    "hover_joints": None,
    "grasp_joints": None,
    "box_hover_joints": None,
    "box_drop_joints": None,
    "grasp_pixel": None,
    "grasp_area": None,
    "gain_matrix": None,
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"配置已保存: {CONFIG_PATH}")


def detect_cube(frame_bgr: np.ndarray, cfg: dict):
    """返回 (centroid_x, centroid_y, area) 或 None。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(cfg["hsv_low1"]), np.array(cfg["hsv_high1"]))
    m2 = cv2.inRange(hsv, np.array(cfg["hsv_low2"]), np.array(cfg["hsv_high2"]))
    mask = cv2.morphologyEx(m1 | m2, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, cfg["min_area"]
    for c in contours:
        a = cv2.contourArea(c)
        if a >= best_area:
            best, best_area = c, a
    if best is None:
        return None
    x, y, w, h = cv2.boundingRect(best)
    return (x + w / 2.0, y + h / 2.0, best_area, mask)


def open_wrist_cam() -> cv2.VideoCapture:
    # V4L2 后端对物理摄像头最快（FFMPEG 后端仅 ~6.5fps @1080p），先解析符号链接
    # 左臂腕相机走 /dev/left_wrist_cam (video2)
    dev = os.path.realpath("/dev/left_wrist_cam")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("打不开 /dev/left_wrist_cam")
    time.sleep(0.5)
    return cap


def connect_robot() -> XLerobot:
    cfg = XLerobotConfig(use_head=False, use_base=False)
    cfg.cameras = {
        "right_wrist": OpenCVCameraConfig(
            index_or_path="/dev/left_wrist_cam", fps=30, width=640, height=480,
            color_mode=ColorMode.RGB, rotation=Cv2Rotation.NO_ROTATION,
            backend=Cv2Backends.V4L2, fourcc="MJPG",
        )
    }
    robot = XLerobot(cfg)
    # 跳过 connect() 里的交互式回车提示（自动恢复已有标定）
    orig_input = builtins.input
    builtins.input = lambda *a, **k: ""
    try:
        robot.connect()
    finally:
        builtins.input = orig_input
    return robot


def joints_from_obs(obs: dict) -> np.ndarray:
    return np.array([float(obs[k]) for k in JOINT_KEYS])


def read_arm_joints(robot: XLerobot) -> np.ndarray:
    """只读左臂关节角（不碰相机）。取放阶段相机挂了也不能中断走位。"""
    obs = robot.bus1.sync_read("Present_Position", robot.left_arm_motors)
    return np.array([float(obs[f"left_arm_{j}"]) for j in JOINTS])


def read_gripper_present(robot: XLerobot) -> float:
    obs = robot.bus1.sync_read("Present_Position", ["left_arm_gripper"])
    return float(obs["left_arm_gripper"])


def snapshot(robot, cap, path: str) -> None:
    """存一帧腕相机图像，供事后检查。"""
    try:
        frame = grab_frame(robot, cap)
        cv2.imwrite(path, frame)
    except RuntimeError:
        pass


def grab_fresh_frame(robot, cap=None, tries: int = 3) -> np.ndarray:
    """取"新鲜"帧: 相机读线程死后 get_observation 会返回不动的旧帧,
    连拍两帧比对, 内容几乎不变就重连相机再取。"""
    frame = None
    for _ in range(tries):
        f1 = grab_frame(robot, cap)
        time.sleep(0.2)
        f2 = grab_frame(robot, cap)
        diff = np.abs(f1.astype(np.int16) - f2.astype(np.int16)).sum()
        if diff > 2000:
            return f2
        print("  [相机帧疑似停滞, 重连相机...]")
        try:
            cam = robot.cameras.get(CAM_KEY)
            if cam is not None:
                if getattr(cam, "is_connected", False):
                    cam.disconnect()
                time.sleep(0.3)
                cam.connect()
        except Exception as e:
            print(f"  相机重连失败: {e}")
            time.sleep(1.0)
    return f2


_RECORDER = None


def rec_start(path: str = "/tmp/s2b_run.mp4") -> None:
    global _RECORDER
    _RECORDER = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))


def rec_frame(frame: np.ndarray) -> None:
    if _RECORDER is not None:
        _RECORDER.write(frame)


def make_action(joints: np.ndarray, gripper: float) -> dict:
    a = {k: float(v) for k, v in zip(JOINT_KEYS, joints)}
    a[GRIPPER_KEY] = float(gripper)
    return a


def move_to(robot: XLerobot, goal: np.ndarray, gripper: float, cfg: dict, label: str = ""):
    """小步慢速插值到目标关节角。"""
    cur = read_arm_joints(robot)
    n = int(math.ceil(np.abs(goal - cur).max() / cfg["move_step_deg"]))
    n = max(n, 1)
    for i in range(1, n + 1):
        target = cur + (goal - cur) * (i / n)
        robot.send_action(make_action(target, gripper))
        time.sleep(cfg["move_dt"])
        print(f"\r  {label} 插值 {i}/{n}", end="", flush=True)
    print()
    time.sleep(0.3)


def grab_frame(robot: XLerobot, cap=None) -> np.ndarray:
    if cap is not None:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("读取腕部相机失败")
        rec_frame(frame)
        return frame
    last_err = None
    for _ in range(3):
        try:
            obs = robot.get_observation()
            img = obs.get(CAM_KEY)
            if img is not None:
                frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                rec_frame(frame)
                return frame
        except Exception as e:  # USB 瞬断/读线程死亡 -> 重连相机再试
            last_err = e
            print(f"\n相机读取失败（{e}），重连中...")
            time.sleep(0.5)
            try:
                cam = robot.cameras.get(CAM_KEY)
                if cam is not None:
                    if getattr(cam, "is_connected", False):
                        cam.disconnect()
                    cam.connect()
            except Exception as e2:
                print(f"相机重连失败: {e2}")
                time.sleep(1.0)
    raise RuntimeError(f"相机连续读取失败: {last_err}")


def show_and_check(frame: np.ndarray, mask=None, win: str = "square2box") -> bool:
    """显示画面, 返回 False 表示用户按了 q（紧急停止）。"""
    vis = frame.copy()
    if mask is not None:
        vis[mask > 0] = (0, 255, 0)
    cv2.imshow(win, vis)
    return cv2.waitKey(1) & 0xFF != ord("q")


def cmd_preview(cfg: dict) -> None:
    cap = open_wrist_cam()
    print("预览中，绿色=检测到的方块区域，按 q 退出")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取失败，重试...")
                time.sleep(0.2)
                continue
            det = detect_cube(frame, cfg)
            if det is not None:
                cx, cy, area, mask = det
                cv2.drawMarker(frame, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(frame, f"area={area:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                show_and_check(frame, mask)
            else:
                cv2.putText(frame, "NO CUBE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if not show_and_check(frame):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def cmd_calibrate(cfg: dict) -> None:
    robot = connect_robot()
    win = "calibrate"
    try:
        # 标定全程手动摆姿态，左臂必须先下电力矩
        robot.bus1.disable_torque()
        time.sleep(0.3)
        obs = robot.get_observation()
        print("\n== 标定开始（全程不给力矩，用手摆姿态）==")

        input("1/7 用手把夹爪完全张开，按回车记录张开值... ")
        cfg["gripper_open"] = float(robot.get_observation()[GRIPPER_KEY])

        input("2/7 用手把夹爪合拢到能夹住方块的程度，按回车记录闭合值... ")
        cfg["gripper_closed"] = float(robot.get_observation()[GRIPPER_KEY])

        input("3/7 摆到 [悬停在方块上方] 且腕部相机里方块清晰居中，按回车... ")
        obs = robot.get_observation()
        cfg["hover_joints"] = joints_from_obs(obs).tolist()
        det = detect_cube(grab_frame(robot), cfg)
        if det is None:
            raise RuntimeError("看不到方块！请调好 HSV（先跑 preview）或重新摆位")
        cfg["grasp_pixel"] = [det[0], det[1]]
        cfg["grasp_area"] = det[2]
        show_and_check(grab_frame(robot), det[3], win)

        input("4/7 垂直下移到 [夹爪正好套住方块] 的抓取深度，按回车... ")
        cfg["grasp_joints"] = joints_from_obs(robot.get_observation()).tolist()

        input("5/7 摆到 [悬停在盒子上方]，按回车... ")
        cfg["box_hover_joints"] = joints_from_obs(robot.get_observation()).tolist()

        input("6/7 下移到 [盒子里方 releases 高度]，按回车... ")
        cfg["box_drop_joints"] = joints_from_obs(robot.get_observation()).tolist()

        input(
            "7/7 准备探测增益矩阵：请先用手把机械臂放回初始位，\n"
            "    确认手已离开机械臂后按回车（之后会通电并【缓慢】移到悬停位）... "
        )

        print("探测伺服增益矩阵（通电，小幅移动，请离机械臂远一点）...")
        hover = np.array(cfg["hover_joints"])
        probe = {j: JOINTS.index(j) for j in cfg["servo_joints"]}
        J = np.zeros((2, len(probe)))
        for col, (jname, idx) in enumerate(probe.items()):
            robot.bus1.enable_torque(robot.left_arm_motors)
            base = hover.copy()
            move_to(robot, base, cfg["gripper_open"], cfg, "通电缓慢移到悬停位")
            time.sleep(0.4)
            d = np.zeros(len(JOINTS))
            det0 = detect_cube(grab_frame(robot), cfg)
            if det0 is None:
                raise RuntimeError(f"探测 {jname} 前看不到方块")
            d[idx] = cfg["probe_delta_deg"]
            robot.send_action(make_action(base + d, cfg["gripper_open"]))
            time.sleep(0.8)
            det1 = detect_cube(grab_frame(robot), cfg)
            if det1 is None:
                raise RuntimeError(f"移动 {jname} 后看不到方块，减小 probe_delta_deg 再试")
            J[:, col] = [(det1[0] - det0[0]), (det1[1] - det0[1])]
            print(f"  {jname}: +{cfg['probe_delta_deg']}° -> 像素位移 "
                  f"({J[0, col]:.1f}, {J[1, col]:.1f})")
        robot.bus1.disable_torque(robot.left_arm_motors)
        cfg["gain_matrix"] = J.tolist()

        save_config(cfg)
        print("\n标定完成！可以运行: python square2box.py run")
    finally:
        try:
            robot.bus1.disable_torque(robot.left_arm_motors)
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        cv2.destroyAllWindows()


def servo_to_cube(robot, cap, cfg, t0) -> np.ndarray:
    """视觉伺服到方块正上方（对准 grasp_pixel），返回收敛后的实际关节角。"""
    hover = np.array(cfg["hover_joints"])
    target_px = np.array(cfg["grasp_pixel"])
    J_pinv = np.linalg.pinv(np.array(cfg["gain_matrix"]))
    servo_idx = [JOINTS.index(j) for j in cfg["servo_joints"]]

    goal = hover.copy()
    it = 0
    lost = 0
    while it < cfg["max_servo_iters"]:
        if time.time() - t0 > cfg["total_timeout_s"]:
            raise TimeoutError("本轮总超时，中止")
        frame = grab_fresh_frame(robot, cap)
        det = detect_cube(frame, cfg)
        if det is None:
            lost += 1
            print(f"\r  伺服 {it}: 看不到方块 ({lost})", end="", flush=True)
            if lost >= 10:
                raise RuntimeError("连续丢目标，中止")
            if lost == 3:
                # 悬停位看不到方块(可能被放偏了), 左右扫视找
                found = False
                for dp in (-6, -3, 3, 6, 0):
                    goal = hover.copy()
                    goal[JOINTS.index("shoulder_pan")] += dp
                    robot.send_action(make_action(goal, cfg["gripper_open"]))
                    time.sleep(0.5)
                    d = detect_cube(grab_fresh_frame(robot, cap), cfg)
                    if d is not None:
                        print(f"\n  扫视 pan{dp:+d}° 找到方块，继续伺服")
                        found = True
                        break
                if not found:
                    goal = hover.copy()
                    robot.send_action(make_action(goal, cfg["gripper_open"]))
                    time.sleep(0.4)
            time.sleep(0.2)
            continue
        lost = 0
        cx, cy, area, mask = det
        err = np.array([cx, cy]) - target_px
        if not show_and_check(frame, mask):
            raise KeyboardInterrupt("用户按 q 停止")
        if np.abs(err).max() < cfg["pixel_tol"]:
            print(f"\n  伺服收敛: 误差 ({err[0]:.0f},{err[1]:.0f}) px, 迭代 {it} 次")
            return goal
        djoint = np.zeros(len(JOINTS))
        delta = J_pinv @ (-err)
        delta = np.clip(delta, -cfg["servo_step_deg"], cfg["servo_step_deg"])
        for k, idx in enumerate(servo_idx):
            djoint[idx] = delta[k]
        goal = goal + djoint
        it += 1
        print(f"\r  伺服 {it}: 误差 ({err[0]:.0f},{err[1]:.0f}) px, area={area:.0f}",
              end="", flush=True)
        if robot is not None:
            robot.send_action(make_action(goal, cfg["gripper_open"]))
        time.sleep(0.25)
    raise RuntimeError(f"伺服 {cfg['max_servo_iters']} 次未收敛，中止（不抓取）")


def detect_jaw_tips(frame):
    """检测张开的两指指尖, 返回中点 (mx, my); 失败返回 None。

    指隙影像位置随 wrist_roll 绕光轴旋转 (实测 roll -45° 时指尖闭合点从
    (425,410) 移到 (310,330)), 固定标定值只对标定时的腕角有效,
    所以每次下探后现场实测: 两指白色指身是画面底部最亮的两个区域, 各取最高点。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    m = ((V > 215) & (S < 60)).astype(np.uint8) * 255
    m[:300, :] = 0  # 指尖只在画面下部
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 1500]
    if len(cnts) < 2:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    (x1, y1), (x2, y2) = [c[c[:, :, 1].argmin()][0] for c in cnts]
    if abs(x1 - x2) < 25 or abs(x1 - x2) > 560:  # 两指尖间距合理性 (roll 大时透视会收窄)
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def servo_at_depth(robot, cap, cfg, pose, t0) -> np.ndarray:
    """下探到底后在抓取深度现场探测并修正，把方块对到两指之间，再闭爪。

    下探时臂画弧线，悬停位对准不等于深度处对准；且"前后"误差只有 elbow 能修。
    做法: 深度处对 pan/lift/elbow 各做 ±2° 探测得到 2x3 像素雅可比，
    用伪逆解算像素误差 -> 关节修正，迭代到方块进入指间 (grasp_depth_pixel)。
    """
    gap = None
    for _ in range(3):  # 单帧可能运动模糊/遮挡, 多测几帧
        try:
            gap = detect_jaw_tips(grab_fresh_frame(robot, cap))
        except RuntimeError:
            gap = None
        if gap is not None:
            break
        time.sleep(0.15)
    if gap is not None:
        base_target = np.array([gap[0], gap[1] + cfg.get("grasp_dy_px", 80.0)])
        print(f"  指隙实测: ({gap[0]:.0f},{gap[1]:.0f}) + 深度偏移 -> 目标 ({base_target[0]:.0f},{base_target[1]:.0f}) (标定值 {cfg['grasp_depth_pixel']})")
    else:
        print("  指隙检测失败, 用标定值")
        base_target = np.array(cfg["grasp_depth_pixel"])
    idxs = [JOINTS.index(j) for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")]
    goal = pose.copy()
    dcfg = dict(cfg, min_area=250)  # 深度处方块只露一部分, 放宽面积阈值

    def _det():
        try:
            return detect_cube(grab_fresh_frame(robot, cap), dcfg)
        except RuntimeError:
            return None

    def _settle(tol=0.8, timeout=1.6):
        # 等关节真正到位再测: 0.4s 固定等待不够, 电机滞后会让探测/修正
        # 读到中间状态, 迭代叠加指令后一齐生效, 一次冲过目标上百像素。
        end = time.time() + timeout
        while time.time() < end:
            if np.abs(read_arm_joints(robot) - goal).max() < tol:
                return True
            time.sleep(0.08)
        return False

    det0 = _det()
    if det0 is None:
        snapshot(robot, cap, "/tmp/s2b_depth_lost.jpg")
        # 到底后看不到方块: 方块掉出画面下沿/视野外。
        # 组合扫描找回: 抬臂(视野前移) + 腕部下压(wrist+ 使方块影像上移,
        # 实测约 -5px/°)。找到后只修横向 x, 再压回抓取深度闭爪。
        # 实测: 全深度+方块偏下时闭爪才能夹住; 抬着对准像素目标反而抓空。
        wf = JOINTS.index("wrist_flex")
        lift = JOINTS.index("shoulder_lift")
        for du, dw in ((2, 0), (0, 4), (4, 0), (0, 8), (6, 0), (0, 12), (8, 0), (0, 16)):
            if time.time() - t0 > cfg["total_timeout_s"]:
                break
            p = goal.copy()
            p[lift] -= du  # lift 减小 = 抬高
            p[wf] += dw    # wrist+ = 相机下压, 方块影像上移
            robot.send_action(make_action(p, cfg["gripper_open"]))
            _settle()
            d = _det()
            snapshot(robot, cap, f"/tmp/s2b_lift_{du}_{dw}.jpg")
            if d is not None:
                print(f"  抬臂{du}°/腕压{dw}° 找回方块 ({d[0]:.0f},{d[1]:.0f})px")
                goal = p
                det0 = d
                break  # 不压回, 保持该姿态继续走下方雅可比对准
        if det0 is None:
            print("  深度对准: 抬臂/腕压后仍看不到方块")
            return None

    # ---- 现场探测 2x4 雅可比 (pan/lift/elbow/wrist_flex) ----
    J = np.zeros((2, len(idxs)))
    for col, idx in enumerate(idxs):
        p = goal.copy()
        p[idx] += 2.0
        robot.send_action(make_action(p, cfg["gripper_open"]))
        _settle()
        d1 = _det()
        p[idx] -= 4.0
        robot.send_action(make_action(p, cfg["gripper_open"]))
        _settle()
        d2 = _det()
        robot.send_action(make_action(goal, cfg["gripper_open"]))
        _settle()
        if d1 is not None and d2 is not None:
            J[:, col] = [(d1[0] - d2[0]) / 4.0, (d1[1] - d2[1]) / 4.0]
        print(f"  深度探测 {JOINTS[idx]}: 像素/° = ({J[0, col]:.1f}, {J[1, col]:.1f})")

    if np.abs(J).max() < 1.0:
        # 探测值接近 0 说明该姿态下跟踪不可靠，伪逆会给出乱跳的修正
        print("  深度探测退化(像素响应过小)，跳过修正直接闭爪")
        return goal

    # ---- 伪逆迭代修正 ----
    # 腕部只用于"看"(找回方块时的 tilt), 不参与修正:
    #   1) wrist 响应最大, 伪逆会走这条"捷径"大幅改写腕角, 姿态被带偏;
    #   2) 恢复腕角的影像平移补偿基于 ±2° 探测的线性外推, tilt 大了不成立
    #      (实测恢复腕角后方块仍停在补偿目标处, 标准抓取点完全落空)。
    wf_idx = JOINTS.index("wrist_flex")
    wf_col = idxs.index(wf_idx)
    wf_tilt = goal[wf_idx] - pose[wf_idx]
    target = base_target + J[:, wf_col] * wf_tilt
    corr_idxs = [i for i in idxs if i != wf_idx]
    J_pinv = np.linalg.pinv(np.delete(J, wf_col, axis=1))
    det = None
    for it in range(10):
        if time.time() - t0 > cfg["total_timeout_s"]:
            break
        det = _det()
        if det is None:
            snapshot(robot, cap, "/tmp/s2b_depth_lost_fix.jpg")
            print("  深度修正中丢失方块，放弃本次闭爪（回悬停位重新伺服）")
            return None
        err = np.array([det[0], det[1]]) - target
        if np.abs(err).max() < 12:
            print(f"  深度对准收敛: 误差 ({err[0]:.0f},{err[1]:.0f}) px, 迭代 {it} 次")
            break
        delta = np.clip(J_pinv @ (-err) * 0.8, -3.0, 3.0)
        for k, idx in enumerate(corr_idxs):
            goal[idx] += delta[k]
        robot.send_action(make_action(goal, cfg["gripper_open"]))
        _settle()
    else:
        if det is not None:
            print(f"  深度对准未完全收敛: 误差 ({det[0]-target[0]:.0f},{det[1]-target[1]:.0f}) px")
    if det is None:
        return None
    if np.abs(np.array([det[0], det[1]]) - target).max() > 25.0:
        print(f"  深度对准偏差过大 ({det[0]-target[0]:.0f},{det[1]-target[1]:.0f}) px，放弃本次闭爪")
        return None

    # 恢复腕角后必须复检: 补偿是线性近似, 对不准就绝不闭爪
    goal[wf_idx] -= wf_tilt
    robot.send_action(make_action(goal, cfg["gripper_open"]))
    _settle()
    if wf_tilt != 0.0:
        d = _det()
        snapshot(robot, cap, "/tmp/s2b_depth_verify.jpg")
        std_target = base_target
        if d is None or np.abs(np.array([d[0], d[1]]) - std_target).max() > 25:
            got = "看不到方块" if d is None else f"({d[0]:.0f},{d[1]:.0f})px 偏离标准抓取点"
            print(f"  恢复腕角后复检失败: {got}，放弃本次闭爪")
            return None

    # z 向说明: 目标 y 即 z 控制 —— 方块影像贴近画面下沿 = 指尖平齐方块高度。
    # 不要再额外下压: 下压会让方块影像上移, 反而越压越远。
    return goal


def cube_image_angle(det):
    """方块边缘相对画面水平轴的夹角, 折叠到 [-45,45)。

    cv2 5.0 的 minAreaRect 角度范围是 (-90,0], 用模运算折叠,
    正方形对称性保证折叠后误差不超过 45°。
    """
    cnts, _ = cv2.findContours(det[3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), ang = cv2.minAreaRect(c)
    return ((ang + 45.0) % 90.0) - 45.0


def align_cube_roll(robot, cap, cfg, t0):
    """闭爪前把方块转正: 斜角闭爪会被爪垫弹开, 夹爪压到桌面(实测空合 16.8° 被压到 11.1°)。

    现场探测 wrist_roll 的影像角度响应斜率, 按斜率转腕。
    指隙影像位置随腕角变化, 由 servo_at_depth 里的 detect_jaw_tips 现场实测,
    转腕只负责把方块转正, 两者解耦。
    """
    roll = JOINTS.index("wrist_roll")
    probe = 8.0
    det0 = detect_cube(grab_frame(robot, cap), cfg)
    if det0 is None:
        return
    a0 = cube_image_angle(det0)
    if a0 is None or abs(a0) < 10.0:
        return
    base = read_arm_joints(robot)
    q = base.copy()
    q[roll] += probe
    robot.send_action(make_action(q, cfg["gripper_open"]))
    time.sleep(0.8)
    det1 = detect_cube(grab_frame(robot, cap), cfg)
    robot.send_action(make_action(base, cfg["gripper_open"]))
    time.sleep(0.5)
    if det1 is None:
        print(f"  转腕探测中方块丢失, 跳过转腕 (偏角 {a0:.0f}°)")
        return
    a1 = cube_image_angle(det1)
    if a1 is None:
        return
    slope = (a1 - a0) / probe
    if abs(slope) < 0.05:
        return
    corr = float(np.clip(-a1 / slope, -45.0, 45.0))
    q = base.copy()
    q[roll] += corr
    robot.send_action(make_action(q, cfg["gripper_open"]))
    time.sleep(0.8)
    det2 = detect_cube(grab_frame(robot, cap), cfg)
    a2 = cube_image_angle(det2) if det2 is not None else None
    if a2 is not None and abs(a2) < abs(a1):
        print(f"  转腕对准: 方块偏角 {a0:.0f}° -> {a2:.0f}°, wrist_roll {corr:+.1f}°")
    else:
        robot.send_action(make_action(base, cfg["gripper_open"]))
        time.sleep(0.5)
        print(f"  转腕未改善 (偏角 {a0:.0f}°), 保持原腕角")


def cmd_run(cfg: dict, dry_run: bool, max_cycles: int) -> None:
    need = ["gripper_open", "gripper_closed", "hover_joints", "grasp_joints",
            "box_hover_joints", "box_drop_joints", "grasp_pixel", "gain_matrix"]
    missing = [k for k in need if cfg.get(k) in (None, "")]
    if missing:
        raise SystemExit(f"缺少标定数据 {missing}，请先运行: python square2box.py calibrate")

    hover = np.array(cfg["hover_joints"])
    grasp = np.array(cfg["grasp_joints"])
    box_hover = np.array(cfg["box_hover_joints"])
    box_drop = np.array(cfg["box_drop_joints"])

    cap = None
    robot = None
    if dry_run:
        cap = open_wrist_cam()
        print("DRY-RUN: 只做视觉，不动电机。按 q 退出")
    else:
        robot = connect_robot()
        print("已连接，先记录当前姿态作为休息位")

    grasp_thresh = None
    if not dry_run:
        # 启动时测量夹爪"空合值"（无方块合拢到底的读数），用于判定是否夹住
        home0 = read_arm_joints(robot)
        robot.send_action(make_action(home0, cfg["gripper_closed"]))
        time.sleep(cfg["gripper_settle_s"])
        air_closed = read_gripper_present(robot)
        gc = cfg["gripper_closed"]
        if gc > air_closed + 4.0:
            grasp_thresh = (air_closed + gc) / 2.0
        else:
            grasp_thresh = air_closed + 2.0
        robot.send_action(make_action(home0, cfg["gripper_open"]))
        time.sleep(cfg["gripper_settle_s"])
        print(f"夹爪空合值={air_closed:.1f}°, 夹住判定阈值={grasp_thresh:.1f}°")
        # 闭爪不留缝: 压到空合值以下, 靠短时堵转把方块夹紧
        close_cmd = max(0.0, air_closed - 6.0)
        cfg["gripper_closed"] = close_cmd

    cycle = 0
    if not dry_run:
        rec_start()
    try:
        if not dry_run:
            home = read_arm_joints(robot)
            move_to(robot, hover, cfg["gripper_open"], cfg, "移向悬停位")

        while True:
            cycle += 1
            t0 = time.time()
            tag = f"{cycle}/{max_cycles}" if max_cycles > 0 else str(cycle)
            print(f"\n===== 第 {tag} 轮：抓取 → 放入盒子 =====")

            if dry_run:
                print("DRY-RUN 结束")
                return

            # ---- 抓取（带验证+重试）----
            # 关键1: 下探目标 = 伺服收敛后的实际姿态 + 标定的垂直下降增量,
            #   保留伺服的水平修正（固定 grasp_joints 会把修正丢掉导致抓空）
            # 关键2: 夹住后绝不松爪；验证以视觉为准——抬回悬停位后只要桌面
            #   区域还能看到方块(含被撞飞的)就没夹住, 完全看不到才算夹住
            # 注意: 夹爪电机读数不可靠(空合值会漂移, 海绵方块挤压读数重叠), 仅打印参考
            descent_delta = grasp - hover
            try:
                servo_pose = servo_to_cube(robot, cap, cfg, t0)
            except (RuntimeError, TimeoutError) as e:
                print(f"\n桌面上找不到方块（{e}），停止")
                break
            grasped = False
            for attempt in range(1, int(cfg["grasp_retries"]) + 1):
                align_cube_roll(robot, cap, cfg, t0)
                grasp_goal = servo_pose + descent_delta
                move_to(robot, grasp_goal, cfg["gripper_open"], cfg,
                        f"下探抓取(第{attempt}次)")
                grasp_goal = servo_at_depth(robot, cap, cfg, grasp_goal, t0)
                if grasp_goal is None:
                    print("  ✗ 深度处看不到方块，回悬停位重新伺服")
                    robot.send_action(make_action(hover, cfg["gripper_open"]))
                    time.sleep(0.3)
                    servo_pose = servo_to_cube(robot, cap, cfg, t0)
                    continue
                snapshot(robot, cap, f"/tmp/s2b_before_close_{attempt}.jpg")
                cfg["gripper_closed"] = close_cmd  # 每次闭爪都重新满力压紧
                robot.send_action(make_action(grasp_goal, cfg["gripper_closed"]))
                time.sleep(cfg["gripper_settle_s"])
                present = read_gripper_present(robot)
                snapshot(robot, cap, f"/tmp/s2b_close_depth_{attempt}.jpg")
                # 安全: 搬运期间改用保持力矩(当前读数-2°), 避免长时间堵转过热
                cfg["gripper_closed"] = max(close_cmd, present - 2.0)
                # 夹住状态抬回悬停位做视觉验证（全程不松爪）
                pre_lift = grasp_goal.copy()
                pre_lift[1] -= 10.0
                move_to(robot, pre_lift, cfg["gripper_closed"], cfg, "垂直预抬升")
                move_to(robot, hover, cfg["gripper_closed"], cfg, "抬回悬停位")
                snapshot(robot, cap, f"/tmp/s2b_after_close_{attempt}.jpg")
                on_table = False
                try:
                    det = detect_cube(grab_fresh_frame(robot, cap), cfg)
                except RuntimeError:
                    det = None
                # 抬到悬停位后: 桌面区域(画面上部)还能看到方块 = 没夹住
                # (方块可能被撞到别处, 所以看整个桌面区域, 不只原位置)
                if det is not None and det[1] < cfg["table_band_y"]:
                    on_table = True
                # 夹住需双证据: 抬升后桌面看不到方块, 且夹爪读数明显高于空合值
                # (方块被撞出视野时空合读数≈11.7°, 真夹住海绵会撑住 13.4-14°)
                if not on_table and present >= close_cmd + 1.5:
                    grasped = True
                    print(f"  ✓ 已夹住方块 (夹爪读数 {present:.1f}°, 桌面无方块)")
                    break
                if not on_table:
                    print(f"  ⚠ 桌面看不到方块但夹爪 {present:.1f}° 接近空合值，方块可能被撞飞，按未夹住处理")
                print(f"  ✗ 未夹住 (夹爪 {present:.1f}°)，松爪重试")
                robot.send_action(make_action(hover, cfg["gripper_open"]))
                time.sleep(0.3)
                servo_pose = servo_to_cube(robot, cap, cfg, t0)
                # 每次重试稍微下探深一点（shoulder_lift 增大=降低）
                descent_delta[1] += cfg["deepen_per_retry_deg"]
            if not grasped:
                print(f"第 {tag} 轮: {cfg['grasp_retries']} 次都没夹住，跳过放置")
                if max_cycles > 0 and cycle >= max_cycles:
                    break
                continue
            move_to(robot, box_hover, cfg["gripper_closed"], cfg, "移到盒子上方")
            move_to(robot, box_drop, cfg["gripper_closed"], cfg, "下降到盒子")
            robot.send_action(make_action(box_drop, cfg["gripper_open"]))
            time.sleep(cfg["gripper_settle_s"])
            # 松爪后先垂直抬一段再水平离开，防止带倒盒子
            post_lift = box_drop.copy()
            post_lift[1] -= 12.0
            move_to(robot, post_lift, cfg["gripper_open"], cfg, "垂直抬离")
            move_to(robot, box_hover, cfg["gripper_open"], cfg, "回盒子上方")
            move_to(robot, hover, cfg["gripper_open"], cfg, "回悬停位")
            print(f"第 {tag} 轮完成，用时 {time.time()-t0:.0f}s，方块已放入盒子，开始下一轮")

            if max_cycles > 0 and cycle >= max_cycles:
                print(f"\n已跑满 {max_cycles} 轮")
                break
    finally:
        global _RECORDER
        if _RECORDER is not None:
            _RECORDER.release()
            _RECORDER = None
        if robot is not None:
            try:
                robot.bus1.disable_torque(robot.left_arm_motors)
            except Exception:
                pass
            try:
                robot.disconnect()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("（已断开连接并下电力矩）")


def main() -> None:
    ap = argparse.ArgumentParser(description="左臂抓方块放盒子（腕部相机视觉伺服）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preview", help="只开相机看检测效果")
    sub.add_parser("calibrate", help="示教关键点+探测增益矩阵")
    p_run = sub.add_parser("run", help="循环执行 抓取-放入-取出-放回")
    p_run.add_argument("--dry-run", action="store_true", help="只看视觉不动电机")
    p_run.add_argument("--cycles", type=int, default=None,
                       help="跑几轮后停止（默认读 config 的 max_cycles，0=无限循环）")
    args = ap.parse_args()

    cfg = load_config()
    if getattr(args, "cycles", None) is not None:
        cfg["max_cycles"] = args.cycles
    if args.cmd == "preview":
        cmd_preview(cfg)
    elif args.cmd == "calibrate":
        cmd_calibrate(cfg)
    elif args.cmd == "run":
        cmd_run(cfg, args.dry_run, int(cfg["max_cycles"]))


if __name__ == "__main__":
    main()
