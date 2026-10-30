#!/usr/bin/env python
"""calibrate_all.py — square2box 全量点位标定（交互式）。

流程：
  1. 连接机械臂，卸力，打开相机/YOLO。
  2. 按提示用手摆位，逐项记录：
       gripper_open / gripper_closed
       hover_joints（桌面上方，能看见方块）
       grasp_joints（桌面抓取深度）
       box_hover_joints（盒子上方）
       box_drop_joints（盒内释放高度）
       box_pick_joints（盒内抓取深度）
       table_drop_joints（桌面释放高度）
       box_look_joints（看盒子内部的观察位）
       home_joints（安全休息位）
     可选：safe_joints（夹爪完全自由的姿态，用于测空合值）
  3. 可选自动探测：
       gain_matrix（悬停视觉伺服 2xN 像素增益）
       depth_gain_matrix（抓取深度 2x4 雅可比）
       roll_slope_deg_per_deg（腕部角度响应斜率）
  4. 全部写回 config.json。

用法：
  conda activate hupy3.12
  cd /home/zhoulin/hu/square2box
  python calibrate_all.py            # 交互式全量标定
  python calibrate_all.py --check    # 只读检查，不提示、不动电机
  python calibrate_all.py --no-probe # 只标点位，不做自动增益探测

安全：
  - 手动摆位阶段全程卸力；自动探测阶段会通电并小幅移动，必须确认手已离开。
  - 任何异常/退出都会先卸力再断开。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import square2box_loop as S

CALIB_DIR = S.HERE / "calib_frames"


def ask(prompt: str, default: str = "") -> str:
    try:
        ans = input(prompt).strip()
    except EOFError:
        return default
    return ans or default


def save_frame(camera, detector, name: str, det=None):
    try:
        frame = camera.read()
        if det is None:
            det = detector.detect(frame)
        vis = detector.annotate(frame.copy(), det)
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        path = CALIB_DIR / f"{name}.jpg"
        cv2.imwrite(str(path), vis)
        print(f"  画面已保存: {path}")
    except Exception as e:
        print(f"  保存画面失败: {e}")


def detect_now(camera, detector, roi=None):
    frame = camera.read()
    det = detector.detect(frame, roi)
    return frame, det


def show_det(det):
    if det is None:
        print("  未检测到方块")
    else:
        print(f"  检测: {det.source} conf={det.conf:.2f} "
              f"center=({det.cx:.0f},{det.cy:.0f}) area={det.area:.0f} "
              f"angle={det.angle:.1f}")


def record_pose(robot, cfg, key: str, prompt: str, save=True):
    print()
    print(prompt)
    print("  用手扶住机械臂保持姿态，另一只手按回车记录（输入 s 跳过）...")
    ans = ask("  > ", "ok")
    if ans.lower() == "s":
        print(f"  跳过 {key}")
        return False
    q = robot.read_joints()
    cfg[key] = q.tolist()
    if save:
        S.save_config({key: cfg[key]})
    print(f"  已记录 {key} = {np.round(q, 2).tolist()}")
    return True


def record_gripper(robot, cfg, key: str, prompt: str):
    print()
    print(prompt)
    print("  用手保持夹爪位置，另一只手按回车记录...")
    ask("  > ")
    v = robot.read_gripper()
    cfg[key] = float(v)
    S.save_config({key: float(v)})
    print(f"  已记录 {key} = {v:.2f}")


def record_hover(robot, cfg, camera, detector):
    print()
    print("== 3/10 桌面上方（hover）==")
    print("把机械臂摆到方块正上方，夹爪张开，腕相机里方块清晰可见且大致居中。")
    ok = False
    det = None
    for attempt in range(5):
        ask("  摆好后按回车检测方块... ")
        frame, det = detect_now(camera, detector)
        show_det(det)
        save_frame(camera, detector, "hover", det)
        if det is not None:
            ok = True
            break
        print("  没检测到，调整方块/光照/姿态后重试。")
    if not ok:
        raise SystemExit("hover 位连续 5 次检测不到方块，标定中止")
    cfg["hover_joints"] = robot.read_joints().tolist()
    cfg["grasp_pixel"] = [float(det.cx), float(det.cy)]
    cfg["grasp_area"] = float(det.area)
    S.save_config({
        "hover_joints": cfg["hover_joints"],
        "grasp_pixel": cfg["grasp_pixel"],
        "grasp_area": cfg["grasp_area"],
    })
    print(f"  已记录 hover_joints = {np.round(cfg['hover_joints'], 2).tolist()}")
    print(f"  已记录 grasp_pixel = {cfg['grasp_pixel']}")


def record_depth_pixel(robot, cfg, camera, detector):
    print()
    print("== 可选：抓取深度处的方块像素（grasp_depth_pixel）==")
    ans = ask("  是否现在记录？(y/n，默认 y) ", "y")
    if ans.lower() != "y":
        print("  跳过，保留旧 grasp_depth_pixel")
        return
    frame, det = detect_now(camera, detector)
    show_det(det)
    save_frame(camera, detector, "grasp_depth", det)
    if det is None:
        print("  深度处看不到方块，保留旧值")
        return
    cfg["grasp_depth_pixel"] = [float(det.cx), float(det.cy)]
    S.save_config({"grasp_depth_pixel": cfg["grasp_depth_pixel"]})
    print(f"  已记录 grasp_depth_pixel = {cfg['grasp_depth_pixel']}")

    # 顺带量一下指隙到方块中心的 y 偏移（深度对准用）
    gap = S.Runner.detect_jaw_tips(None, frame)
    if gap is not None:
        dy = float(det.cy - gap[1])
        print(f"  指隙 y={gap[1]:.0f}，方块 y={det.cy:.0f}，建议 grasp_dy_px={dy:.0f}"
              f"（当前 {cfg.get('grasp_dy_px')}）")
        ans = ask("  是否更新 grasp_dy_px？(y/n，默认 y) ", "y")
        if ans.lower() == "y":
            cfg["grasp_dy_px"] = dy
            S.save_config({"grasp_dy_px": dy})
            print(f"  已更新 grasp_dy_px = {dy:.1f}")
    else:
        print("  指隙检测失败，保留旧 grasp_dy_px")


def record_box_target(cfg, camera, detector):
    print()
    print("== 可选：盒内验证 ROI 中心（box_target_pixel）==")
    ans = ask("  是否用当前检测到的方块中心更新 box_target_pixel？(y/n，默认 y) ", "y")
    if ans.lower() != "y":
        print("  跳过，保留旧 box_target_pixel")
        return
    frame, det = detect_now(camera, detector)
    show_det(det)
    save_frame(camera, detector, "box_look", det)
    if det is None:
        print("  盒内没检测到方块，保留旧值")
        return
    cfg["box_target_pixel"] = [float(det.cx), float(det.cy)]
    S.save_config({"box_target_pixel": cfg["box_target_pixel"]})
    print(f"  已记录 box_target_pixel = {cfg['box_target_pixel']}")


def probe_gain(robot, cfg, camera, detector):
    print()
    print("== 自动探测悬停视觉增益 gain_matrix ==")
    print("会通电并让机械臂在 hover 位小幅移动（每个关节 +probe_delta 度）。")
    ans = ask("  确认手已离开机械臂，按回车开始（输入 n 跳过）... ", "y")
    if ans.lower() == "n":
        print("  跳过 gain_matrix")
        return
    hover = S.require_pose(cfg, "hover_joints")
    open_cmd = float(cfg["gripper_open"])
    delta = float(cfg.get("probe_delta_deg", 3.0))
    servo_joints = list(cfg["servo_joints"])
    robot.enable_torque()
    try:
        robot.set_gripper(open_cmd, settle=0.35)
        robot.move_joints(hover, open_cmd, speed=10, label="标定 hover")
        time.sleep(0.4)
        _, det0 = detect_now(camera, detector)
        if det0 is None:
            print("  hover 位看不到方块，跳过 gain_matrix")
            return
        J = np.zeros((2, len(servo_joints)))
        for col, jname in enumerate(servo_joints):
            idx = S.JOINTS.index(jname)
            p = hover.copy()
            p[idx] += delta
            robot.move_joints(p, open_cmd, speed=8, label=f"探测 {jname}")
            time.sleep(0.35)
            _, det1 = detect_now(camera, detector)
            robot.move_joints(hover, open_cmd, speed=8, label="回 hover")
            time.sleep(0.35)
            if det1 is None:
                print(f"  {jname}: 移动后丢失方块，跳过该列")
                continue
            J[:, col] = det1.center - det0.center
            print(f"  {jname}: +{delta:.1f}° -> 像素位移 "
                  f"({J[0, col]:.2f}, {J[1, col]:.2f})")
        if float(np.abs(J).max()) < 0.5:
            print("  增益过小，未保存")
            return
        cfg["gain_matrix"] = J.tolist()
        S.save_config({"gain_matrix": J.tolist()})
        print("  已保存 gain_matrix:")
        print(np.round(J, 3))
    finally:
        robot.disable_torque()


def probe_depth(robot, cfg, camera, detector):
    print()
    print("== 自动探测抓取深度雅可比 depth_gain_matrix ==")
    print("会通电并让机械臂在 grasp 深度小幅移动（pan/lift/elbow/wrist 各 ±2°）。")
    ans = ask("  确认手已离开机械臂，按回车开始（输入 n 跳过）... ", "y")
    if ans.lower() == "n":
        print("  跳过 depth_gain_matrix")
        return
    base = S.require_pose(cfg, "grasp_joints")
    open_cmd = float(cfg["gripper_open"])
    idxs = [S.JOINTS.index(j) for j in
            ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")]
    robot.enable_torque()
    try:
        robot.set_gripper(open_cmd, settle=0.35)
        robot.move_joints(base, open_cmd, speed=10, label="标定 grasp 深度")
        time.sleep(0.4)
        _, d0 = detect_now(camera, detector)
        if d0 is None:
            print("  抓取深度看不到方块，跳过 depth_gain_matrix")
            return
        J = np.zeros((2, len(idxs)))
        for col, idx in enumerate(idxs):
            p = base.copy()
            p[idx] += 2.0
            robot.move_joints(p, open_cmd, speed=8, label=f"探测 {S.JOINTS[idx]}")
            time.sleep(0.30)
            _, d1 = detect_now(camera, detector)
            p[idx] -= 4.0
            robot.move_joints(p, open_cmd, speed=8, label=f"探测 {S.JOINTS[idx]}")
            time.sleep(0.30)
            _, d2 = detect_now(camera, detector)
            robot.move_joints(base, open_cmd, speed=8, label="回 grasp 深度")
            time.sleep(0.30)
            if d1 is None or d2 is None:
                print(f"  {S.JOINTS[idx]}: 探测丢失，跳过")
                continue
            J[:, col] = [(d1.cx - d2.cx) / 4.0, (d1.cy - d2.cy) / 4.0]
            print(f"  {S.JOINTS[idx]}: 像素/度 = ({J[0, col]:.2f}, {J[1, col]:.2f})")
        if float(np.abs(J).max()) < 1.0:
            print("  响应过小，未保存")
            return
        cfg["depth_gain_matrix"] = J.tolist()
        S.save_config({"depth_gain_matrix": J.tolist()})
        print("  已保存 depth_gain_matrix:")
        print(np.round(J, 3))
    finally:
        robot.disable_torque()


def probe_roll(robot, cfg, camera, detector):
    print()
    print("== 自动探测腕部角度斜率 roll_slope_deg_per_deg ==")
    ans = ask("  确认手已离开机械臂，按回车开始（输入 n 跳过）... ", "y")
    if ans.lower() == "n":
        print("  跳过 roll_slope")
        return
    base = S.require_pose(cfg, "grasp_joints")
    open_cmd = float(cfg["gripper_open"])
    roll = S.JOINTS.index("wrist_roll")
    robot.enable_torque()
    try:
        robot.set_gripper(open_cmd, settle=0.35)
        robot.move_joints(base, open_cmd, speed=10, label="标定 roll")
        time.sleep(0.4)
        _, d0 = detect_now(camera, detector)
        if d0 is None:
            print("  看不到方块，跳过 roll_slope")
            return
        p = base.copy()
        p[roll] += 8.0
        robot.move_joints(p, open_cmd, speed=8, label="转腕 +8°")
        time.sleep(0.35)
        _, d1 = detect_now(camera, detector)
        robot.move_joints(base, open_cmd, speed=8, label="回腕角")
        time.sleep(0.30)
        if d1 is None:
            print("  转腕后丢失方块，跳过")
            return
        slope = (d1.angle - d0.angle) / 8.0
        if abs(slope) < 0.05:
            print(f"  角度响应过小（{slope:.3f}），未保存")
            return
        cfg["roll_slope_deg_per_deg"] = float(slope)
        S.save_config({"roll_slope_deg_per_deg": float(slope)})
        print(f"  已保存 roll_slope = {slope:.3f} deg/deg")
    finally:
        robot.disable_torque()


def cmd_check(cfg):
    print("=== 当前 config.json 关键值 ===")
    for k in ("gripper_open", "gripper_closed", "hover_joints", "grasp_joints",
              "box_hover_joints", "box_drop_joints", "box_pick_joints",
              "table_drop_joints", "box_look_joints", "home_joints",
              "grasp_pixel", "grasp_depth_pixel", "gain_matrix",
              "depth_gain_matrix", "roll_slope_deg_per_deg",
              "servo_joints", "measure_pan_offset"):
        print(f"  {k}: {cfg.get(k)}")
    print("\n只读检查：不连接机械臂、不动电机。")
    cam = S.CameraStream(logger=S.logging.getLogger("s2b"))
    det = S.Detector(cfg, S.logging.getLogger("s2b"))
    cam.start()
    try:
        time.sleep(0.8)
        frame, d = detect_now(cam, det)
        show_det(d)
        save_frame(cam, det, "check", d)
    finally:
        cam.stop()


def main():
    ap = argparse.ArgumentParser(description="square2box 全量点位标定")
    ap.add_argument("--check", action="store_true", help="只读检查，不提示、不动电机")
    ap.add_argument("--no-probe", action="store_true", help="只标点位，不做自动增益探测")
    args = ap.parse_args()

    cfg = S.load_config()
    logger = S.setup_logging(cfg)
    if args.check:
        cmd_check(cfg)
        return

    print("=" * 70)
    print("square2box 全量点位标定")
    print("手动摆位阶段全程卸力；自动探测阶段会通电小幅移动。")
    print("请确保机械臂周围无障碍，手能随时断开电源。")
    print("=" * 70)

    robot = S.Robot(cfg, S.threading.Event(), logger)
    cam = S.CameraStream(logger=logger)
    det = S.Detector(cfg, logger)
    try:
        robot.connect()
        robot.disable_torque()
        time.sleep(0.3)
        cam.start()
        time.sleep(0.8)

        print("\n== 1/10 夹爪完全张开 ==")
        record_gripper(robot, cfg, "gripper_open",
                       "用手把夹爪完全张开（不要用力掰）")
        print("\n== 2/10 夹爪夹住方块 ==")
        record_gripper(robot, cfg, "gripper_closed",
                       "用手把夹爪合拢到刚好夹住方块的程度")

        record_hover(robot, cfg, cam, det)

        print("\n== 4/10 桌面抓取深度 ==")
        print("保持夹爪张开，用手把机械臂下移到夹爪正好套住方块的位置。")
        record_pose(robot, cfg, "grasp_joints", "摆到桌面抓取深度。")
        record_depth_pixel(robot, cfg, cam, det)

        print("\n== 5/10 盒子上方 ==")
        print("把机械臂移到盒子上方，夹爪张开，确认不会碰盒壁。")
        record_pose(robot, cfg, "box_hover_joints", "摆到盒子上方。")

        print("\n== 6/10 盒内释放高度 ==")
        print("下移到盒子内部、方块能稳定落下的释放高度。")
        record_pose(robot, cfg, "box_drop_joints", "摆到盒内释放高度。")

        print("\n== 7/10 盒内抓取深度 ==")
        print("把夹爪套住盒子里的方块（如果盒里暂时没有方块，可先放一个再摆）。")
        record_pose(robot, cfg, "box_pick_joints", "摆到盒内抓取深度。")

        print("\n== 8/10 桌面释放高度 ==")
        print("把机械臂移到桌面放方块的位置，下移到释放高度。")
        record_pose(robot, cfg, "table_drop_joints", "摆到桌面释放高度。")

        print("\n== 9/10 盒子观察位 ==")
        print("摆到能看清盒子内部的观察位（略高、腕部略下压）。")
        record_pose(robot, cfg, "box_look_joints", "摆到盒子观察位。")
        record_box_target(cfg, cam, det)

        print("\n== 10/10 安全休息位 ==")
        print("摆到一个远离桌面和盒子的安全姿态，用于恢复/回位。")
        record_pose(robot, cfg, "home_joints", "摆到安全休息位。")

        print("\n== 可选：夹爪自由空间姿态 safe_joints ==")
        ans = ask("  是否记录一个夹爪完全自由、不碰任何东西的姿态？(y/n，默认 n) ", "n")
        if ans.lower() == "y":
            record_pose(robot, cfg, "safe_joints",
                        "摆到夹爪完全悬空、不碰桌面/盒子/线缆的姿态。")

        if not args.no_probe:
            print("\n" + "=" * 70)
            print("点位记录完成。下面进入自动探测（会通电小幅移动）。")
            print("=" * 70)
            probe_gain(robot, cfg, cam, det)
            probe_depth(robot, cfg, cam, det)
            probe_roll(robot, cfg, cam, det)
            # 探测结束后回安全位再卸力，避免在抓取深度处断电下垂
            try:
                home = cfg.get("home_joints") or cfg.get("hover_joints")
                if home:
                    robot.enable_torque()
                    robot.set_gripper(float(cfg["gripper_open"]), settle=0.35)
                    robot.move_joints(np.asarray(home, float),
                                      float(cfg["gripper_open"]),
                                      speed=20, label="标定结束回位")
            finally:
                robot.disable_torque()
        else:
            print("\n--no-probe：跳过自动增益探测，保留旧 gain_matrix/depth_gain_matrix/roll_slope。")

        print("\n标定完成，config.json 已更新。")
        print("建议先跑: python square2box_loop.py self-test")
        print("再跑一轮: python square2box_loop.py run --cycles 1")
    finally:
        try:
            robot.disable_torque()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
