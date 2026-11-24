#!/usr/bin/env python
"""run_calibrated_flow.py — 标定后的双臂点位巡演（无摄像头）。

用 calibrate2arm.py 刚标定好的 config.json，按固定顺序走每个点位：

    home -> carry -> 桌面 approach/hover/pick
         -> 合爪（模拟抓起）
         -> 桌面 hover -> carry -> 盒内 approach/hover/pick/drop
         -> 开爪（模拟放下）
         -> 盒内 hover/approach -> carry -> home

只依赖 armcore / armdriver，不导入 armvision、cv2 或 ultralytics，
所以运行时不打开任何摄像头。

用法（conda activate hupy3.12）：
    cd /home/zhoulin/hu/square2box/1
    python run_calibrated_flow.py --dry-run       # 只看动作序列，不连硬件
    python run_calibrated_flow.py --check         # 只读校验，不动电机
    python run_calibrated_flow.py                 # 双臂各巡演一轮
    python run_calibrated_flow.py --arm left --speed 0.4
    python run_calibrated_flow.py --cycles 3
    python run_calibrated_flow.py --minutes 5
    python run_calibrated_flow.py --home-only

安全：
    * 启动前清理工作区、手离开机械臂；首次实跑建议 --speed 0.4。
    * Ctrl+C 会先开爪、再卸力、最后断开；臂停留在当前姿态，不会自动回 home。
    * 到位超时/残差过大时立即中止，不再继续下一个点位。
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import armcore as C
from armdriver import (
    DualArm,
    FeetechMotorsBus,
    calibration_from_bus,
    make_motors,
    resolve_arm_ports,
)

HERE = Path(__file__).resolve().parent

# 每条臂的三段式动作。grip 表示这一段执行时夹爪目标：
# open = gripper_open，closed = gripper_closed。
SEGMENT_TABLE = ["home", "carry", "table_approach", "table_hover", "table_pick"]
SEGMENT_TO_BOX = [
    "table_pick", "table_hover", "carry",
    "box_approach", "box_hover", "box_pick", "box_drop",
]
SEGMENT_HOME = ["box_drop", "box_hover", "box_approach", "carry", "home"]


def arm_list(name: str) -> list[str]:
    return ["left", "right"] if name == "both" else [name]


def build_ops(grasp: bool, grip_dwell: float) -> list[dict]:
    """生成一条臂的巡演操作序列。"""
    ops: list[dict] = [
        {"kind": "move", "keys": SEGMENT_TABLE, "grip": "open",
         "label": "桌面下探段"},
    ]
    if grasp:
        ops.append({"kind": "grip", "which": "closed", "dwell": grip_dwell,
                    "label": "桌面合爪"})
    ops.append({"kind": "move", "keys": SEGMENT_TO_BOX,
                "grip": "closed" if grasp else "open", "label": "送盒段"})
    if grasp:
        ops.append({"kind": "grip", "which": "open", "dwell": grip_dwell,
                    "label": "盒内开爪"})
    ops.append({"kind": "move", "keys": SEGMENT_HOME, "grip": "open",
                "label": "回 home 段"})
    return ops


def gripper_value(cfg: dict, arm: str, which: str) -> float:
    value = cfg["arms"][arm].get(f"gripper_{which}")
    if value is None:
        raise SystemExit(
            f"[{arm}] 缺少 gripper_{which}，请先运行: "
            f"python ../2arm/calibrate2arm.py --arm {arm}")
    return float(value)


def read_firmware(ports: dict[str, str], arms: list[str], logger) -> dict[str, dict]:
    """只读电机固件标定（不上力、不写参数）。"""
    firmware: dict[str, dict] = {}
    for arm in arms:
        bus = None
        try:
            bus = FeetechMotorsBus(
                port=ports[arm], motors=make_motors(C.ARM_PREFIX[arm]))
            bus.connect(handshake=False)
            firmware[arm] = calibration_from_bus(bus)
            logger.info("[%s] 固件标定读取成功（%s）", arm, ports[arm])
        except Exception as exc:
            logger.warning("[%s] 读固件标定失败: %s", arm, exc)
        finally:
            if bus is not None:
                try:
                    bus.disconnect(disable_torque=False)
                except Exception:
                    pass
    return firmware


def fingerprint_report(cfg: dict,
                        firmware: dict[str, dict]) -> tuple[list[str], list[str]]:
    """比对每个点位的标定指纹与当前固件。返回 (阻断问题, 警告)。"""
    problems: list[str] = []
    warnings: list[str] = []
    for arm, fw in firmware.items():
        missing = C.missing_poses(cfg, arm)
        differs, unknown = C.pose_calib_mismatch(cfg, arm, fw)
        differs = [k for k in differs if k not in missing]
        unknown = [k for k in unknown if k not in missing]
        if differs:
            problems.append(
                f"[{arm}] 这些点位是在旧电机标定下录的，不能直接使用: "
                f"{', '.join(differs)}\n"
                f"  请重跑: python ../2arm/calibrate2arm.py --arm {arm}")
        if unknown:
            warnings.append(
                f"[{arm}] 这些点位没有标定指纹，无法确认是否属于当前电机标定: "
                f"{', '.join(unknown)}")
        warnings.extend(C.tight_pose_report(cfg, arm, fw))
    return problems, warnings


def validate_config(cfg: dict, arms: list[str]) -> tuple[list[str], list[str]]:
    """返回 (阻断问题, 警告)。阻断问题存在时不允许驱动电机。"""
    problems: list[str] = []
    warnings: list[str] = []

    for arm in arms:
        missing = C.missing_poses(cfg, arm)
        if missing:
            problems.append(
                f"[{arm}] 缺少 {len(missing)} 个位姿: {', '.join(missing)}\n"
                f"  请先运行: python ../2arm/calibrate2arm.py --arm {arm}")
        for which in ("open", "closed"):
            try:
                gripper_value(cfg, arm, which)
            except SystemExit as exc:
                problems.append(str(exc))

        problems.extend(C.pose_saturation_report(cfg, arm))
        warnings.extend(C.duplicate_pose_report(cfg, arm))
        warnings.extend(C.pose_delta_report(cfg, arm))

    return problems, warnings


def print_plan(cfg: dict, arms: list[str], grasp: bool, grip_dwell: float) -> None:
    print("=" * 70)
    print("标定后无摄像头巡演（--dry-run：只打印计划，不连接硬件）")
    print("=" * 70)
    for arm in arms:
        print(f"\n[{arm}] 夹爪开={gripper_value(cfg, arm, 'open'):.1f} "
              f"合={gripper_value(cfg, arm, 'closed'):.1f}")
        for i, op in enumerate(build_ops(grasp, grip_dwell), 1):
            if op["kind"] == "move":
                print(f"  {i}. 夹爪 {op['grip']:<6}  {' -> '.join(op['keys'])}")
            else:
                print(f"  {i}. {op['label']}  （目标 "
                      f"{gripper_value(cfg, arm, op['which']):.1f}，"
                      f"停留 {op['dwell']:.2f}s）")
    print("\n提示：实跑时两臂按顺序执行，左臂完成后才动右臂。")


class TourRunner:
    """把 config.json 里的标定位姿按固定序列发给 DualArm。"""

    def __init__(self, cfg: dict, dual: DualArm, logger,
                 speed: float, step_pause: float, grasp: bool,
                 grip_dwell: float):
        self.cfg = cfg
        self.dual = dual
        self.log = logger
        self.speed = float(speed)
        self.step_pause = float(step_pause)
        self.grasp = bool(grasp)
        self.grip_dwell = float(grip_dwell)
        self.open_g = {a: gripper_value(cfg, a, "open") for a in C.ARMS}
        self.close_g = {a: gripper_value(cfg, a, "closed") for a in C.ARMS}

    def _pose(self, arm: str, key: str):
        return C.get_pose(self.cfg, arm, key)

    def _move(self, arm: str, keys: list[str], gripper: float,
              label: str, speed_scale: float) -> None:
        a = self.dual.arms[arm]
        waypoints = [self._pose(arm, key) for key in keys]
        est = a.start(
            waypoints,
            gripper=gripper,
            label=f"{arm} {label}",
            speed_scale=self.speed * max(0.05, float(speed_scale)),
        )
        ok = a.wait(timeout=est + 15.0)
        if not ok:
            raise RuntimeError(
                f"[{arm}] {label} 未到位（超时或残差超过 "
                f"motion.arrive_tol={self.cfg['motion']['arrive_tol']}）")
        self.log.info("[%s] %s 完成 %.1fs  %s -> %s", arm, label,
                      a.last_motion_s, keys[0], keys[-1])
        if self.step_pause > 0:
            time.sleep(self.step_pause)

    def _grip(self, arm: str, which: str) -> None:
        value = self.close_g[arm] if which == "closed" else self.open_g[arm]
        target = self.dual.arms[arm].set_gripper(value)
        self.log.info("[%s] 夹爪%s：目标 %.1f，读数 %.1f",
                      arm, "合" if which == "closed" else "开", value, target)
        if self.grip_dwell > 0:
            time.sleep(self.grip_dwell)

    def temp_scale(self) -> float:
        """按 safety 阈值返回速度倍率；达到停止温度则抛异常。"""
        try:
            temp = float(self.dual.max_temp())
        except Exception as exc:
            self.log.debug("读温度失败，按常温处理: %s", exc)
            return 1.0
        stop = float(self.cfg["safety"]["temp_stop_c"])
        slow = float(self.cfg["safety"]["temp_slow_c"])
        warn = float(self.cfg["safety"]["temp_warn_c"])
        self.log.info("当前最高电机温度 %.1f°C", temp)
        if temp >= stop:
            raise RuntimeError(f"电机温度 {temp:.1f}°C 达到停止阈值 {stop:.0f}°C")
        if temp >= slow:
            self.log.warning("电机温度 %.1f°C >= %.0f°C，本轮降速到 0.7×",
                             temp, slow)
            return 0.7
        if temp >= warn:
            self.log.warning("电机温度 %.1f°C 偏高（警告阈值 %.0f°C）",
                             temp, warn)
        return 1.0

    def tour_arm(self, arm: str, speed_scale: float) -> None:
        self.log.info("=" * 60)
        self.log.info("[%s] 开始点位巡演（段速度 %.2f）", arm, speed_scale)
        for op in build_ops(self.grasp, self.grip_dwell):
            if op["kind"] == "move":
                gripper = (self.open_g[arm] if op["grip"] == "open"
                           else self.close_g[arm])
                self._move(arm, op["keys"], gripper, op["label"], speed_scale)
            else:
                self._grip(arm, op["which"])
        self.log.info("[%s] 巡演完成，已回到 home（张爪）", arm)

    def go_home(self, arm: str, speed_scale: float) -> None:
        self._move(arm, ["home"], self.open_g[arm], "回 home", speed_scale)


def cmd_check(cfg: dict, arms: list[str], logger) -> int:
    """只读校验：配置完整性 + 电机固件标定指纹。全程不上力、不动电机。"""
    print("=" * 70)
    print("只读校验（--check：不使能力矩，不发送运动指令）")
    print("=" * 70)

    problems, warnings = validate_config(cfg, arms)

    firmware: dict[str, dict] = {}
    try:
        ports = resolve_arm_ports(cfg, logger)
    except Exception as exc:
        print(f"\n串口解析失败，跳过固件指纹检查: {exc}")
        ports = None

    if ports:
        firmware = read_firmware(ports, arms, logger)
        for arm in arms:
            if arm not in firmware:
                warnings.append(f"[{arm}] 固件标定读取失败，跳过该臂指纹检查")
        fp_problems, fp_warnings = fingerprint_report(cfg, firmware)
        problems.extend(fp_problems)
        warnings.extend(fp_warnings)

    for line in warnings:
        print(line)
    if problems:
        print("\n发现阻断问题：")
        for line in problems:
            print(line)
        print(f"\n校验失败：{len(problems)} 个阻断问题。")
        return 1
    print("\n校验通过：位姿齐全，未发现阻断问题。")
    return 0


def shutdown(dual: DualArm | None, logger) -> None:
    """统一退出：停轨迹线程 -> 开爪 -> 卸力 -> 断开。"""
    if dual is None:
        return
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except Exception:
        pass
    try:
        dual.stop_workers()
    except Exception as exc:
        logger.debug("停止轨迹线程失败: %s", exc)
    for arm in C.ARMS:
        # 用当前关节读数作为目标位置：轨迹线程被停下后，若沿用 _planned_q
        # 会把臂重新朝旧终点拉过去，卸力前又产生一段意外运动。
        try:
            q = dual.arms[arm].read_joints()
            value = float(dual.cfg["arms"][arm]["gripper_open"])
            dual.arms[arm].bus.write_goal(q, value)
        except Exception as exc:
            logger.debug("[%s] 退出开爪失败: %s", arm, exc)
    try:
        dual.disable_torque()
    except Exception as exc:
        logger.debug("退出卸力失败: %s", exc)
    try:
        dual.disconnect()
    except Exception as exc:
        logger.debug("退出断开失败: %s", exc)
    logger.info("退出清理完成（已连接的臂会开爪、卸力并断开）。")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="标定后的双臂点位巡演（无摄像头，按固定位姿序列运动）")
    ap.add_argument("--arm", choices=("left", "right", "both"), default="both",
                    help="执行哪条臂（默认 both，两条按顺序）")
    ap.add_argument("--cycles", type=int, default=0,
                    help="巡演轮数；0 且未给 --minutes 时默认 1 轮")
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="持续运行多少分钟；0=不按时间限制")
    ap.add_argument("--speed", type=float, default=0.6,
                    help="速度倍率（0.05~2.0，默认 0.6；首跑建议 0.4）")
    ap.add_argument("--step-pause", type=float, default=0.3,
                    help="每段动作结束后的停留秒数")
    ap.add_argument("--grip-dwell", type=float, default=0.5,
                    help="合爪/开爪后的停留秒数")
    ap.add_argument("--cycle-pause", type=float, default=0.3,
                    help="两轮之间的停留秒数")
    ap.add_argument("--no-grasp", action="store_true",
                    help="全程张爪，只走位姿，不做合爪/开爪动作")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印动作序列，不连接硬件")
    ap.add_argument("--check", action="store_true",
                    help="只读校验位姿和固件指纹，不使能力矩")
    ap.add_argument("--home-only", action="store_true",
                    help="只把选中的臂开爪回 home 后退出")
    ap.add_argument("--config", default=None,
                    help="config.json 路径（默认 1/config.json）")
    args = ap.parse_args()

    if not (0.05 <= args.speed <= 2.0):
        ap.error("--speed 必须在 0.05 ~ 2.0 之间")
    if args.cycles < 0 or args.minutes < 0:
        ap.error("--cycles / --minutes 不能为负数")

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    arms = arm_list(args.arm)
    logger = C.setup_logging(cfg, name="calibrated_flow")

    problems, warnings = validate_config(cfg, arms)

    if args.dry_run:
        if problems:
            print("标定配置不完整或不可用：", file=sys.stderr)
            for line in problems:
                print(line, file=sys.stderr)
            return 2
        print_plan(cfg, arms, not args.no_grasp, args.grip_dwell)
        if warnings:
            print("\n配置提示（不阻断运行）：")
            for line in warnings:
                print(line)
        return 0

    if args.check:
        return cmd_check(cfg, arms, logger)

    for line in warnings:
        logger.warning("%s", line)
    if problems:
        for line in problems:
            logger.error("%s", line)
        print("标定配置不完整或不可用，已拒绝启动。", file=sys.stderr)
        return 2

    max_cycles = int(args.cycles)
    minutes = float(args.minutes)
    if max_cycles <= 0 and minutes <= 0:
        max_cycles = 1
    deadline = time.time() + minutes * 60.0 if minutes > 0 else None

    stop_event = threading.Event()

    def on_signal(signum, _frame):
        logger.warning("收到信号 %s，准备停止", signum)
        stop_event.set()
        raise C.StopRequested()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    dual: DualArm | None = None
    ok = False
    try:
        dual = DualArm(cfg, logger, stop_event)

        # 上力之前先只读固件指纹：归一化坐标系变了以后，旧点位会落到
        # 完全不同的物理角度，必须在任何运动指令之前拦住。
        firmware = read_firmware(dual.ports, arms, logger)
        fp_problems, fp_warnings = fingerprint_report(cfg, firmware)
        for line in fp_warnings:
            logger.warning("%s", line)
        missing_fw = [a for a in arms if a not in firmware]
        if missing_fw:
            for arm in missing_fw:
                logger.error("[%s] 无法读取固件标定，拒绝运动", arm)
            print("无法确认标定位姿属于当前电机标定，已拒绝运动。",
                  file=sys.stderr)
            return 2
        if fp_problems:
            for line in fp_problems:
                logger.error("%s", line)
            print("标定位姿与当前电机标定不一致，已拒绝运动；"
                  "请先重跑 calibrate2arm.py 并同步 config.json。",
                  file=sys.stderr)
            return 2

        dual.connect(only=arms)
        runner = TourRunner(cfg, dual, logger, args.speed, args.step_pause,
                            not args.no_grasp, args.grip_dwell)

        if args.home_only:
            scale = runner.temp_scale()
            for arm in arms:
                runner.go_home(arm, scale)
            ok = True
            logger.info("已回 home。")
            return 0

        idx = 0
        while True:
            if max_cycles and idx >= max_cycles:
                break
            if deadline is not None and idx > 0 and time.time() >= deadline:
                break
            idx += 1
            scale = runner.temp_scale()
            logger.info("# 第 %d 轮开始（左/右顺序执行）", idx)
            t0 = time.perf_counter()
            for arm in arms:
                runner.tour_arm(arm, scale)
            logger.info("# 第 %d 轮完成，用时 %.1fs", idx, time.perf_counter() - t0)
            if args.cycle_pause > 0:
                time.sleep(args.cycle_pause)

        ok = True
        logger.info("巡演结束：共 %d 轮。", idx)
        return 0
    except C.StopRequested:
        logger.warning("收到停止请求")
        return 130
    except Exception as exc:
        logger.error("运行失败: %s: %s", type(exc).__name__, exc)
        logger.debug(traceback.format_exc())
        return 1
    finally:
        shutdown(dual, logger)
        logger.info("退出%s", "正常" if ok else "（未完成）")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断，退出清理会开爪并卸力。")
        sys.exit(130)
