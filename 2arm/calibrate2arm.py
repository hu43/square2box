#!/usr/bin/env python
"""calibrate2arm.py — 双臂点位标定（交互式，手摆 + 回车记录）。

流程：
  1. 连接机械臂，卸力，打开腕相机和 YOLO。
  2. 按提示用手摆位，逐项记录：
       夹爪完全张开 -> gripper_open
       夹爪夹住方块 -> gripper_closed
       每条臂 10 个位姿（桌面/盒子各 5 个，含观察位和抓取位）
  3. 全部写回 config.json，并顺手算出视觉对准点和盒内检测范围。

用法：
  conda activate hupy3.12
  cd /home/zhoulin/hu/square2box/2arm

  python calibrate2arm.py               # 左臂、右臂依次标定
  python calibrate2arm.py --arm left    # 只标左臂
  python calibrate2arm.py --arm right   # 只标右臂
  python calibrate2arm.py --check       # 只读检查，不动电机

  只重标某几个点：
  python calibrate2arm.py --arm left --only table_hover,table_pick

安全：
  - 手摆阶段全程卸力，可以随意掰动。
  - 摆好按回车即可；t 可选（加力保持，方便松手）；记完自动卸力。
  - 任何异常 / Ctrl+C / q 退出都会先卸力再断开。
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
from armdriver import DualArm, load_arm_calibration, probe_port_counts, resolve_arm_ports
from armvision import CameraStream, Detector

CALIB_DIR = C.CALIB_DIR


def ask(prompt: str, default: str = "") -> str:
    try:
        ans = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return ans or default


def cam_device(cfg: dict, arm: str) -> str:
    d = cfg["devices"]
    usb = d.get(f"{arm}_camera_usb")
    if usb:
        got = C.find_video_by_usb(str(usb))
        if got:
            return got
    primary = str(d.get(f"{arm}_camera") or "")
    for c in [primary] + list(d.get(f"{arm}_camera_fallbacks") or []):
        if c and Path(str(c)).exists():
            return str(c)
    return primary


def save_frame(cam, det: Detector, name: str, det_result) -> None:
    try:
        frame = cam.fresh(max_age=1.5)
        if frame is None:
            frame = cam.read(timeout=1.0)
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        path = CALIB_DIR / f"{name}.jpg"
        cv2.imwrite(str(path), det.annotate(frame.copy(), det_result))
        print(f"    画面: {path}")
    except Exception as exc:
        print(f"    保存画面失败: {exc}")


# ---------------------------------------------------------------------------
# 只读检查
# ---------------------------------------------------------------------------

def _check_pose_calib(cfg: dict) -> int:
    """只读：确认每个点位都属于当前电机标定。返回问题数。"""
    from armdriver import calibration_from_bus, make_motors
    from lerobot.motors.feetech import FeetechMotorsBus

    try:
        ports = resolve_arm_ports(cfg)
    except Exception as exc:
        print(f"  串口解析失败（跳过）: {exc}")
        return 0
    problems = 0
    for arm in ("left", "right"):
        motors = make_motors(C.ARM_PREFIX[arm])
        bus = None
        try:
            bus = FeetechMotorsBus(port=ports[arm], motors=motors)
            bus.connect(handshake=False)
            fw = calibration_from_bus(bus)
        except Exception as exc:
            print(f"  {arm}: 读固件标定失败（跳过）: {exc}")
            problems += 1
            continue
        finally:
            if bus is not None:
                try:
                    bus.disconnect(disable_torque=False)
                except Exception:
                    pass
        differs, unknown = C.pose_calib_mismatch(cfg, arm, fw)
        missing = C.missing_poses(cfg, arm)
        differs = [k for k in differs if k not in missing]
        unknown = [k for k in unknown if k not in missing]
        if differs or unknown:
            if differs:
                print(f"  {arm}: 这些点位不是当前电机标定下录的:")
                print(f"        {', '.join(differs)}")
                print(f"        多半是标定中途退出，导致新旧坐标混在一起。")
            if unknown:
                print(f"  {arm}: 这些点位没留标定指纹（加指纹之前录的）:")
                print(f"        {', '.join(unknown)}")
                print(f"        无法确认是否属于当前电机标定。")
            print(f"        重标: python calibrate2arm.py --arm {arm}")
            problems += 1
        else:
            print(f"  {arm}: 所有点位都属于当前电机标定")
        tight = C.tight_pose_report(cfg, arm, fw)
        if tight:
            print(f"  {arm}: 这些点位离标定边界不到 5（容易被插值/微调挤出去）:")
            for line in tight:
                print(line)
            print(f"        不阻断运行，但建议把对应关节挪开一点重标。")
    return problems


def _check_calib_matches_firmware(cfg: dict, cals: dict[str, dict]) -> int:
    """只读比对：官方标定文件 vs 电机固件里实际生效的限位。返回问题数。

    归一化一律用官方文件。固件只是执行层：限位退化成 min == max 时电机
    会把关节锁死（发指令不动），这是必须拦下的问题；仅仅不一致会截断
    运动范围，属于该重跑官方标定的警告。
    """
    from armdriver import calibration_from_bus, make_motors
    from lerobot.motors.feetech import FeetechMotorsBus

    try:
        ports = resolve_arm_ports(cfg)
    except Exception:
        return 0
    problems = 0
    for arm in ("left", "right"):
        sub = cals.get(arm)
        if not sub:
            continue
        motors = make_motors(C.ARM_PREFIX[arm])
        bus = None
        try:
            bus = FeetechMotorsBus(port=ports[arm], motors=motors, calibration=sub)
            bus.connect(handshake=False)
            fw = calibration_from_bus(bus)
        except Exception as exc:
            print(f"  {arm}: 读固件限位失败（跳过比对）: {exc}")
            continue
        finally:
            if bus is not None:
                try:
                    bus.disconnect(disable_torque=False)
                except Exception:
                    pass
        degen = [n for n, c in fw.items() if c.range_max - c.range_min < 100]
        diff = [n for n, c in fw.items()
                if n not in degen and n in sub
                and (c.range_min != sub[n].range_min
                     or c.range_max != sub[n].range_max
                     or c.homing_offset != sub[n].homing_offset)]
        if degen:
            print(f"  {arm}: 电机固件里的行程已退化（min==max），关节会被锁死: {degen}")
            for n in degen:
                c = fw[n]
                print(f"        {n}: 固件 {c.range_min}..{c.range_max}，"
                      f"文件 {sub[n].range_min}..{sub[n].range_max}")
            print(f"        按官方标定提示处回车可把文件写回电机: "
                  f"{C.official_calibrate_cmd(arm)}")
            problems += 1
        if diff:
            print(f"  {arm}: 固件限位与标定文件不一致 {diff}")
            print(f"        归一化用文件，但固件限位会截断运动。重跑官方标定并在"
                  f"提示处按回车即可写回: {C.official_calibrate_cmd(arm)}")
            problems += 1
        if not degen and not diff:
            print(f"  {arm}: 固件限位与标定文件一致")
    return problems


def cmd_check(cfg: dict) -> int:
    print("=" * 68)
    print("只读检查（不会动电机）")
    print("=" * 68)
    problems = 0

    print("\n[1] 电机标定文件（官方 lerobot-calibrate 写的）")
    cals: dict[str, dict] = {}
    for arm in ("left", "right"):
        p = C.calibration_paths(cfg).get(arm)
        if p is None:
            print(f"  {arm}: config 里没配电机标定路径")
            problems += 1
            continue
        try:
            cals[arm] = load_arm_calibration(p, arm)
            print(f"  {arm}: OK {p}（{len(cals[arm])} 个条目）")
        except SystemExit as exc:
            print(f"  {arm}: {exc}")
            problems += 1
    if cals:
        problems += _check_calib_matches_firmware(cfg, cals)

    print("\n[2] 串口")
    counts = probe_port_counts()
    for p, n in sorted(counts.items()):
        print(f"  {p}: {n} 个电机" + ("" if n >= 6 else "  <-- 不足 6 个"))
    try:
        ports = resolve_arm_ports(cfg)
        print(f"  left={ports['left']}  right={ports['right']}")
    except Exception as exc:
        print(f"  {exc}")
        problems += 1

    print("\n[3] 位姿")
    for arm in ("left", "right"):
        missing = C.missing_poses(cfg, arm)
        a = C.arm_cfg(cfg, arm)
        if missing:
            print(f"  {arm}: 缺 {len(missing)} 个 -> {', '.join(missing)}")
            problems += 1
        else:
            print(f"  {arm}: 10 个位姿齐全")
        if a.get("gripper_open") is None or a.get("gripper_closed") is None:
            print(f"  {arm}: 夹爪开合未标定")
            problems += 1
        sat = C.pose_saturation_report(cfg, arm)
        if sat:
            print(f"  {arm}: 这些位姿读数顶到 ±100（被截断），必须重标")
            for line in sat:
                print(line)
            problems += 1
        same = C.duplicate_pose_report(cfg, arm)
        if same:
            print(f"  {arm}: 这些点位与参考位重合（快速段没有高度差）")
            for line in same:
                print(line)
            problems += 1
        if not (a.get("gains") or {}).get("box_roi"):
            print(f"  {arm}: box_roi 未设置（盒内检测会用全画面）")
            problems += 1

    print("\n[4] 点位与电机标定的对应关系")
    problems += _check_pose_calib(cfg)

    print()
    print("检查通过。" if problems == 0 else f"发现 {problems} 个问题。")
    return 0 if problems == 0 else 1


# ---------------------------------------------------------------------------
# 标定
# ---------------------------------------------------------------------------

_JOINT_LABEL = {
    "shoulder_pan": "pan",
    "shoulder_lift": "lift",
    "elbow_flex": "elbow",
    "wrist_flex": "wflex",
    "wrist_roll": "wroll",
}


class Session:
    """一条臂的标定会话：机械臂 + 腕相机 + 检测器。"""

    def __init__(self, cfg: dict, logger, arm: str):
        self.cfg = cfg
        self.arm_name = arm
        self.stop_event = threading.Event()
        self.dual = DualArm(cfg, logger, self.stop_event)
        self.cam: CameraStream | None = None
        self.det: Detector | None = None
        self.poses: dict[str, list[float]] = {}
        self.pose_calib: dict[str, str] = {}
        self.torque_on = False

    def __enter__(self):
        # 只连要标的这条臂：另一条线松了也不影响。
        self.dual.connect(only=[self.arm_name])
        self.cam = CameraStream(cam_device(self.cfg, self.arm_name),
                                logger=None, name=self.arm_name,
                                width=self.cfg["devices"]["camera_width"],
                                height=self.cfg["devices"]["camera_height"],
                                fps=self.cfg["devices"]["camera_fps"])
        self.cam.start()
        self.det = Detector(self.cfg, None)
        time.sleep(0.4)
        return self

    def __exit__(self, *exc):
        self.free()
        if self.cam is not None:
            self.cam.stop()
        try:
            self.dual.disconnect()
        except Exception:
            pass
        return False

    @property
    def arm(self):
        return self.dual.arms[self.arm_name]

    def free(self) -> None:
        self.torque_on = False
        try:
            self.arm.bus.disable_torque()
        except Exception:
            pass

    def set_torque(self, on: bool) -> bool:
        """临时加力。先把目标位置写成当前读数再上力，免得臂跳向旧目标。

        卸力摆位时手一松，臂就塌到机械止点（左臂 elbow_flex 塌到 +100），
        记下来的点在边界上根本没法用。加力后可以先松手，再从容按回车。
        """
        if not on:
            self.free()
            return False
        try:
            bus = self.arm.bus.bus
            if bus is None:
                raise RuntimeError("臂未连接")
            names = self.arm.bus.joint_names + [self.arm.bus.gripper_name]
            pos = bus.sync_read("Present_Position", names)
            bus.sync_write("Goal_Position", {n: float(pos[n]) for n in names})
            self.arm.bus.enable_torque()
        except Exception as exc:
            print(f"  ! 加力失败: {exc}")
            self.free()
            return False
        self.torque_on = True
        return True

    def read(self) -> np.ndarray:
        return self.arm.bus.read_joints()

    def joint_margins(self) -> list[tuple[str, float]]:
        """各关节离标定边界的归一化距离（正数=还有余量，0=已截断）。"""
        try:
            bus = self.arm.bus.bus
            raw = bus.sync_read("Present_Position",
                                self.arm.bus.joint_names + [self.arm.bus.gripper_name],
                                normalize=False)
        except Exception:
            return []
        out = []
        for j in C.JOINTS:
            name = f"{self.arm.bus.prefix}_{j}"
            c = bus.calibration.get(name)
            if c is None:
                continue
            r = float(raw[name])
            span = max(1.0, float(c.range_max - c.range_min))
            low = (r - c.range_min) / span * 200.0
            high = (c.range_max - r) / span * 200.0
            out.append((j, min(low, high)))
        return out

    def margins(self) -> str:
        """一行显示各关节余量，摆位时瞄一眼就能避开饱和。"""
        return "  ".join(f"{_JOINT_LABEL.get(j, j):>5s}={m:4.1f}"
                         + ("!" if m < 1.0 else "")
                         for j, m in self.joint_margins())

    def read_gripper(self) -> float:
        return self.arm.bus.read_gripper()

    def detect(self, roi=None):
        frame = self.cam.fresh(max_age=1.0)
        if frame is None:
            try:
                frame = self.cam.read(timeout=1.0)
            except Exception:
                return None, None
        return frame, self.det.detect(frame, roi=roi, force_yolo=True)

    def record(self, key: str) -> np.ndarray:
        q = self.read()
        self.poses[key] = [round(float(v), 4) for v in q]
        C.set_pose(self.cfg, self.arm_name, key, q)
        # 逐点位记下“录它时用的是哪套电机标定”，防止中途退出后新旧点位混用。
        bus = getattr(self.arm.bus, "bus", None)
        cal = getattr(bus, "calibration", None) if bus is not None else None
        if cal:
            C.set_pose_calib(self.cfg, self.arm_name, key, cal)
            self.pose_calib[key] = C.calib_fingerprint_short(cal)
        return q

    def flush(self) -> None:
        if not self.poses:
            return
        payload: dict = {"poses": self.poses}
        if self.pose_calib:
            payload["pose_calib"] = self.pose_calib
        C.save_config({"arms": {self.arm_name: payload}})


def record_gripper(s: Session, cfg: dict) -> None:
    arm_cfg = C.arm_cfg(cfg, s.arm_name)
    print()
    print("  -- 夹爪 --")
    ask("  用手把夹爪完全张开，按回车记录 gripper_open... ")
    go = s.read_gripper()
    print(f"  gripper_open = {go:.2f}")

    ask("  把夹爪合到刚好夹住绿色方块，按回车记录 gripper_closed... ")
    gc = s.read_gripper()
    print(f"  gripper_closed = {gc:.2f}")
    if gc >= go:
        print("  ! 闭合值不小于张开值，方向可能反了，请检查夹爪安装。")

    arm_cfg["gripper_open"] = float(go)
    arm_cfg["gripper_closed"] = float(gc)
    C.save_config({"arms": {s.arm_name: {"gripper_open": round(float(go), 3),
                                         "gripper_closed": round(float(gc), 3)}}})


def _reference_pose(s: Session, cfg: dict, key: str) -> np.ndarray | None:
    """取一个可比的参考位姿：优先本次会话刚记的，其次当前标定下录的。

    config.json 里可能还留着旧电机标定下录的点位（整臂重标的中途），
    拿它当基准会把新点误判成“不竖直”或“和 hover 重合”，所以只有
    指纹和现在一致时才用。
    """
    if key in s.poses:
        return np.asarray(s.poses[key], dtype=float)
    bus = getattr(s.arm.bus, "bus", None)
    cal = getattr(bus, "calibration", None) if bus is not None else None
    if cal:
        recorded = C.get_pose_calib(cfg, s.arm_name).get(key)
        if recorded and recorded != C.calib_fingerprint_short(cal):
            return None
    return C.get_pose(cfg, s.arm_name, key, required=False)


def pose_notes(s: Session, cfg: dict, key: str, q: np.ndarray) -> list[str]:
    """记录之后给的一行提示。只提示，不拦人。"""
    out: list[str] = []
    clipped = C.saturated_joints(q)
    if clipped:
        out.append(f"读数被截断，高度分不开: {', '.join(clipped)}"
                   f"（可用 --watch 看是机械到头还是行程标窄）")
    tight = [j for j, m in s.joint_margins() if m < 5.0]
    if tight:
        out.append(f"离标定边界不到 5: {', '.join(tight)}")
    pair = {"table_approach": "table_hover", "table_hover": "table_approach",
            "box_approach": "box_hover", "box_hover": "box_approach"}
    if key in pair:
        other = _reference_pose(s, cfg, pair[key])
        if other is not None and float(np.abs(q - other).max()) < 3.0:
            gap = float(np.abs(q - other).max())
            out.append(f"与 {pair[key]} 只差 {gap:.1f}，快速段没有减速余地")
    if key in ("table_pick", "box_pick"):
        hover_key = "table_hover" if key == "table_pick" else "box_hover"
        hover = _reference_pose(s, cfg, hover_key)
        if hover is not None:
            d = q - hover
            if abs(d[0]) > 2.0 or abs(d[4]) > 2.0:
                out.append(f"pan/roll 与 {hover_key} 差 {d[0]:+.1f}/{d[4]:+.1f}，"
                           f"不是竖直下降")
    return out


def record_poses(s: Session, cfg: dict, keys: list[str]) -> dict:
    """按顺序逐项记录位姿：摆好按回车，s 跳过，q 退出。

    和官方 lerobot-calibrate 一样，这里不拦人——记什么就是什么。
    有问题只打一行提示，最后用 --check 汇总。
    """
    dets: dict[str, tuple] = {}
    total = len(keys)
    for i, key in enumerate(keys, start=1):
        title, desc = C.POSE_HELP[key]
        print()
        print(f"  -- {i}/{total}  {key} — {title} --")
        print(f"     {desc}")
        while True:
            ans = ask("     摆好后按回车记录（t 加力 / s 跳过 / q 退出）: ", "")
            key_in = ans.lower()
            if key_in == "q":
                return dets
            if key_in == "s":
                print(f"   跳过 {key}")
                break
            if key_in == "t":
                s.set_torque(not s.torque_on)
                print("   已加力，可以松手" if s.torque_on else "   已卸力")
                continue
            q = s.record(key)
            print(f"   已记录 {key} = {C.fmt_q(q, 2)}")
            for note in pose_notes(s, cfg, key, q):
                print(f"   · {note}")
            # 抓取/观察位：顺手看一眼相机能不能看到方块
            if key in ("table_hover", "table_pick", "box_hover", "box_pick", "box_drop"):
                frame, det = s.detect()
                if det is None:
                    print("    相机没检测到方块（不影响记录，但请确认画面里能看到方块）")
                else:
                    print(f"    检测: {det}")
                    dets[key] = (float(det.cx), float(det.cy))
                save_frame(s.cam, s.det, f"{s.arm_name}_{key}", det)

            s.flush()
            if s.torque_on:
                s.set_torque(False)
            break
    return dets


def _roi_around(cx: float, cy: float, w: float = 340.0, h: float = 300.0,
                fw: int = 640, fh: int = 480) -> list[int]:
    return [int(max(0, cx - w / 2)), int(max(0, cy - h / 2)),
            int(min(fw - 1, cx + w / 2)), int(min(fh - 1, cy + h / 2))]


def finish_gains(s: Session, cfg: dict, dets: dict) -> None:
    """用标定时看到的目标位置，算出对准点和盒内检测范围。"""
    gains = dict(C.arm_cfg(cfg, s.arm_name).get("gains") or {})
    changed = False
    if "table_hover" in dets:
        gains["align_px"] = [round(v, 2) for v in dets["table_hover"]]
        changed = True
    box_det = dets.get("box_drop") or dets.get("box_hover")
    if box_det is not None:
        gains["box_roi"] = _roi_around(box_det[0], box_det[1])
        changed = True
    elif C.get_pose(cfg, s.arm_name, "box_hover", required=False) is not None:
        # 盒内暂时没方块时，用画面中心作锚点（观察位正对盒子）
        gains["box_roi"] = _roi_around(320.0, 240.0)
        changed = True
    if changed:
        print()
        print(f"  对准点 align_px = {gains.get('align_px')}")
        print(f"  盒内范围 box_roi = {gains.get('box_roi')}")
        C.save_config({"arms": {s.arm_name: {"gains": gains}}})


def save_calib_fingerprint(s: Session, cfg: dict) -> None:
    """打印本次标定后，哪些点位属于当前电机标定、哪些还是旧的。

    record() 已经逐点记好指纹了，这里只是收尾核对。中途 q / Ctrl+C
    退出时这条信息最重要：剩下的点位仍是旧坐标，不能拿来跑。
    """
    bus = getattr(s.arm.bus, "bus", None)
    cal = getattr(bus, "calibration", None) if bus is not None else None
    if not cal:
        return
    short = C.calib_fingerprint_short(cal)
    recorded = C.get_pose_calib(cfg, s.arm_name)
    ok = [k for k in C.POSE_KEYS
          if C.get_pose(cfg, s.arm_name, k, required=False) is not None
          and recorded.get(k) == short]
    old = [k for k in C.POSE_KEYS
           if C.get_pose(cfg, s.arm_name, k, required=False) is not None
           and recorded.get(k) != short]
    print()
    print(f"  属于当前电机标定的点位: {len(ok)}/{len(C.POSE_KEYS)}")
    if old:
        print(f"  仍是旧标定下录的点位: {', '.join(old)}")
        print(f"  这些点位不能用，请重跑: python calibrate2arm.py --arm {s.arm_name}")


def calibrate_arm(cfg: dict, logger, arm: str, only: list[str] | None) -> int:
    keys = [k for k in C.POSE_KEYS if only is None or k in only]
    if not keys:
        print("没有要标的位姿。")
        return 1
    print()
    print("=" * 68)
    print(f"标定 — {arm}  （{len(keys)} 个位姿）")
    print("=" * 68)
    print("用手摆位，回车记录 / t 加力保持 / s 跳过 / q 退出。")

    with Session(cfg, logger, arm) as s:
        s.free()
        time.sleep(0.3)
        if only is None:
            record_gripper(s, cfg)
        dets = record_poses(s, cfg, keys)
        s.flush()
        if only is None:
            finish_gains(s, cfg, dets)
        save_calib_fingerprint(s, cfg)

        s.free()
        print()
        print(f"[{arm}] 已保存到 {C.CONFIG_PATH}")
        for line in C.pose_delta_report(cfg, arm):
            print(line)

    missing = C.missing_poses(cfg, arm)
    if missing:
        print(f"\n[{arm}] 还有 {len(missing)} 个位姿未标定: {', '.join(missing)}")
        return 1
    bus_cal = None
    try:
        from armdriver import calibration_from_bus, make_motors
        from lerobot.motors.feetech import FeetechMotorsBus
        ports = resolve_arm_ports(cfg)
        bus = FeetechMotorsBus(port=ports[arm], motors=make_motors(C.ARM_PREFIX[arm]))
        try:
            bus.connect(handshake=False)
            bus_cal = calibration_from_bus(bus)
        finally:
            try:
                bus.disconnect(disable_torque=False)
            except Exception:
                pass
    except Exception:
        bus_cal = None
    stale = []
    if bus_cal:
        differs, unknown = C.pose_calib_mismatch(cfg, arm, bus_cal)
        stale = [k for k in differs + unknown if k not in missing]
    if stale:
        print(f"\n[{arm}] 10 个位姿齐全，但 {len(stale)} 个还是旧标定下录的: {', '.join(stale)}")
        print(f"       必须重跑一次（中途不要 q / Ctrl+C）: "
              f"python calibrate2arm.py --arm {arm}")
        return 1
    print(f"\n[{arm}] 10 个位姿齐全，且都属于当前电机标定。")
    return 0


def cmd_watch(cfg: dict, logger, arm: str) -> int:
    """实时显示 raw / 范围 / 归一化，判断关节是到机械止点还是标定扫窄了。

    标定只把“你当时扫过的那段行程”映射成 -100..+100。手推某个关节到推不动，
    看 raw 能不能超过范围上限：
      raw 停在上下限附近就不动了 -> 机械真的到头了，那个姿态不可达
      raw 明显超过上下限         -> 标定时没扫全，重标电机行程
    """
    stop = threading.Event()
    dual = DualArm(cfg, logger, stop)
    try:
        dual.connect(only=[arm])
        a = dual.arms[arm]
        a.bus.disable_torque()
        print()
        print(f"  {arm} 实时读数（全程卸力，可以直接用手推）。Ctrl+C 退出。")
        print("  把关节慢慢推到推不动，看 raw 能不能超过“范围”：")
        print("    raw 停在上/下限附近不动 -> 机械真的到头了，这个姿态不可达")
        print("    raw 明显超出上/下限     -> 标定时没扫全行程，重标电机行程")
        while True:
            bus = a.bus.bus
            if bus is None:
                print("  串口断了。")
                return 1
            names = a.bus.joint_names + [a.bus.gripper_name]
            raw = bus.sync_read("Present_Position", names, normalize=False)
            lines = [f"  {arm}   raw / 标定范围 / 归一化",
                     f"  {'关节':<12}{'raw':>6}  {'范围':>12}  {'归一化':>8}  {'余量':>6}"]
            for j in C.JOINTS:
                c = bus.calibration.get(f"{a.bus.prefix}_{j}")
                if c is None:
                    continue
                r = float(raw[f"{a.bus.prefix}_{j}"])
                span = max(1.0, float(c.range_max - c.range_min))
                norm = (r - c.range_min) / span * 200.0 - 100.0
                margin = min((r - c.range_min) / span * 200.0,
                             (c.range_max - r) / span * 200.0)
                if r > c.range_max:
                    flag = "  超出上限"
                elif r < c.range_min:
                    flag = "  超出下限"
                elif margin < 1.0:
                    flag = "  贴边"
                else:
                    flag = ""
                lines.append(
                    f"  {_JOINT_LABEL.get(j, j):<12}{r:6.0f}  "
                    f"{c.range_min:>5}..{c.range_max:<6}{norm:+8.1f}  "
                    f"{margin:6.1f}{flag}")
            # 终端里原地刷新；重定向到文件时逐帧追加，方便事后看
            clear = "\033[2J\033[H" if sys.stdout.isatty() else ""
            print(clear + "\n".join(lines), flush=True)
            time.sleep(0.15)
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            dual.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="双臂点位标定")
    ap.add_argument("--arm", choices=["left", "right"], help="只标一条臂（默认两条）")
    ap.add_argument("--only", default="", help="只标这些位姿，逗号分隔")
    ap.add_argument("--check", action="store_true", help="只读检查，不动电机")
    ap.add_argument("--watch", action="store_true",
                    help="实时打印 raw/范围/归一化，判断关节是否到机械止点")
    ap.add_argument("--config", default=None, help="config.json 路径")
    args = ap.parse_args()

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="calib2arm")

    if args.check:
        return cmd_check(cfg)

    if args.watch:
        if not args.arm:
            print("--watch 需要指定 --arm left 或 --arm right")
            return 2
        return cmd_watch(cfg, logger, args.arm)

    only = [k.strip() for k in args.only.split(",") if k.strip()] or None
    if only:
        bad = [k for k in only if k not in C.POSE_KEYS]
        if bad:
            print(f"未知位姿: {bad}\n可用: {C.POSE_KEYS}")
            return 2

    rc = 0
    for arm in ([args.arm] if args.arm else ["left", "right"]):
        rc |= calibrate_arm(cfg, logger, arm, only)

    print()
    print("下一步：")
    print(f"  python calibrate2arm.py --check")
    print(f"  python dualarm_loop.py run --cycles 1")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断，已卸力。")
        sys.exit(130)
