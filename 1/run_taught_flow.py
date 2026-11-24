#!/usr/bin/env python
"""run_taught_flow.py — 严格按 14 步示教动作流循环执行（无摄像头）。

从 taught_flow.json 读取 14 步关节值，按索引 1..14 顺序、原样发送：

  1 左臂安全位 -> 2 左臂夹方块前 -> 3 左臂夹方块（合爪）
  -> 4 左臂夹起在中转位 -> 5 左臂放方块前 -> 6 左臂放方块（开爪）
  -> 7 左臂回安全位
  8 右臂安全位 -> 9 右臂夹方块前 -> 10 右臂夹方块（合爪）
  -> 11 右臂夹起在中转位 -> 12 右臂放方块前 -> 13 右臂放方块（开爪）
  -> 14 右臂回安全位 -> 回到第 1 步，循环

位置严格性：
  * 每个动作的目标就是 taught_flow.json 里录下的 5 个关节值；
  * 不叠加释放抬高、pan/深度偏移，也不产生派生 hover；
  * 合爪/开爪只发生在第 3/10、第 6/13 步，按预设 pre/post 固定执行；
  * 重试（--tries > 1）也只是重复同一个步骤、同一个示教点。

用法（conda activate hupy3.12）：
  cd /home/zhoulin/hu/square2box/1
  ./run_taught_flow.sh --check
  ./run_taught_flow.sh --dry-run
  ./run_taught_flow.sh --minutes 5 --speed 0.7
  ./run_taught_flow.sh                 # 无限循环

安全：Ctrl+C 会先开爪、再卸力；方块若还在爪里会落下。
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

import armcore as C
from armdriver import DualArm
from handoff_core import Mover, read_firmware, read_ports, shutdown
from taught_flow_core import (
    FLOW_PATH,
    STEP_DEFS,
    flow_fingerprint_report,
    gripper_values,
    load_flow,
    normalize_flow,
    set_gripper_only,
    step_gripper_text,
    validate_flow,
)

HERE = Path(__file__).resolve().parent


class CycleCSV:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["time", "cycle", "ok", "duration_s", "error"])

    def write(self, cycle: int, ok: bool, duration: float, error: str = "") -> None:
        self.w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), cycle,
                         "ok" if ok else "fail", f"{duration:.1f}",
                         str(error)[:300]])
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


def print_plan(flow: dict, cfg: dict) -> None:
    print("=" * 72)
    print("14 步预设动作流（严格按示教位姿和顺序执行）")
    print("=" * 72)
    for d in STEP_DEFS:
        step = flow["steps"][d["index"] - 1]
        q = step.get("q")
        qtxt = C.fmt_q(q) if q is not None else "未示教"
        post = step.get("post")
        act = (f"{step['pre']} -> {post}" if post else f"保持 {step['pre']}")
        print(f"  {d['index']:>2}. {d['label']:<14} [{d['arm']:<5}] "
              f"夹爪 {act:<12} {qtxt}")
    print()
    print("方块流转约定：")
    print("  - 第 6 步左臂放方块的位置 = 第 10 步右臂夹方块的位置；")
    print("  - 第 13 步右臂放方块的位置 = 下一轮第 3 步左臂夹方块的位置。")
    print("  - 运行时只发送上述示教关节值，不插偏移点；合爪/开爪只在第 3/10、6/13 步。")
    print("  - 启动时默认先预置到第 1/8 步安全位；--no-precondition 可跳过。")


def cmd_check(cfg: dict, logger, flow_path: Path) -> int:
    print("=" * 72)
    print("只读校验（--check：不使能力矩，不发送运动指令）")
    print("=" * 72)
    flow = load_flow(flow_path)
    problems, warnings = validate_flow(flow, cfg, require_complete=True)
    firmware = {}
    try:
        ports = read_ports(cfg, logger)
    except Exception as exc:
        print(f"\n串口解析失败，跳过指纹检查: {exc}")
        ports = None
    if ports:
        firmware = read_firmware(ports, C.ARMS, logger)
        for arm in C.ARMS:
            if arm not in firmware:
                problems.append(f"[{arm}] 固件标定读取失败")
        fp_problems, fp_warnings = flow_fingerprint_report(flow, firmware)
        problems.extend(fp_problems)
        warnings.extend(fp_warnings)

    for line in warnings:
        print(line)
    if problems:
        print("\n阻断问题：")
        for line in problems:
            print(line)
        print(f"\n校验失败：{len(problems)} 个阻断问题。")
        return 1
    print("\n校验通过：14 步齐全，且都属于当前电机标定。")
    return 0


def check_temp(dual: DualArm, cfg: dict, logger, stop_event: threading.Event,
               mover: Mover) -> bool:
    """只在轮次边界调用；不改姿态、不卸力，必要时原地等待降温。"""
    try:
        temp = float(dual.max_temp())
    except Exception as exc:
        logger.debug("读温度失败: %s", exc)
        return True
    slow = float(cfg["safety"]["temp_slow_c"])
    stop = float(cfg["safety"]["temp_stop_c"])
    warn = float(cfg["safety"]["temp_warn_c"])
    logger.info("当前最高电机温度 %.1f°C", temp)
    if temp >= stop:
        wait = float(cfg["safety"]["temp_wait_s"])
        logger.error("温度 %.1f°C 达到停机阈值，原地等待 %.0fs（不改变姿态）",
                     temp, wait)
        end = time.perf_counter() + wait
        while time.perf_counter() < end:
            if stop_event.is_set():
                raise C.StopRequested()
            time.sleep(min(0.5, end - time.perf_counter()))
        try:
            temp = float(dual.max_temp())
        except Exception:
            return True
        if temp >= stop:
            logger.error("等待后温度仍 %.1f°C，退出等待 systemd 重启", temp)
            return False
        logger.info("降温后 %.1f°C，继续", temp)
    if temp >= slow:
        mover.speed_factor = 0.7
        logger.warning("温度 %.1f°C >= %.0f°C，降速到 70%%", temp, slow)
        return True
    mover.speed_factor = 1.0
    if temp >= warn:
        logger.warning("温度 %.1f°C 偏高（警告阈值 %.0f°C）", temp, warn)
    return True


def ensure_bus(dual: DualArm, cfg: dict, logger) -> None:
    threshold = int(cfg["safety"].get("reconnect_after_errors", 6))
    for arm in C.ARMS:
        try:
            streak = int(dual.arms[arm].bus.error_streak)
        except Exception:
            continue
        if streak >= threshold:
            logger.warning("[%s] 总线连续错误 %d 次，轮次边界重连", arm, streak)
            try:
                dual.reconnect(arm)
            except Exception as exc:
                logger.error("[%s] 重连失败: %s", arm, exc)


def run_step(mover: Mover, dual: DualArm, cfg: dict, logger,
             step: dict, gripper_state: dict, load_info: dict,
             tries: int, verify_grasp: bool, grip_dwell: float,
             stop_event: threading.Event) -> tuple[bool, str]:
    """严格执行一个步骤：目标关节值 = step['q']，不做任何修改。"""
    index = step["index"]
    arm = step["arm"]
    label = step["label"]
    pre = step["pre"]
    post = step.get("post")
    q = np.asarray(step["q"], dtype=float)
    open_g, close_g = gripper_values(cfg, arm)
    pre_val = open_g if pre == "open" else close_g
    post_val = open_g if post == "open" else close_g

    last_err = ""
    for attempt in range(1, max(1, int(tries)) + 1):
        if stop_event.is_set():
            raise C.StopRequested()
        # 上一步的结束状态应与本步骤 pre 一致；不一致时只补夹爪动作。
        if gripper_state[arm] != pre:
            try:
                set_gripper_only(dual, cfg, arm, pre)
                gripper_state[arm] = pre
            except Exception as exc:
                last_err = f"设置夹爪 {pre} 失败: {exc}"
                logger.error("[%s] %s", arm, last_err)
                return False, last_err

        ok = mover.mv(arm, [q], gripper=pre_val,
                      label=f"{index} {label}" +
                            (f" (重试 {attempt})" if attempt > 1 else ""),
                      speed=1.0,
                      arrive_tol=(5.0 if post else None))
        if not ok:
            last_err = f"未到位（尝试 {attempt}/{tries}）"
            logger.warning("[%s] %s", arm, last_err)
            if attempt < int(tries):
                continue
            return False, last_err

        if post:
            try:
                set_gripper_only(dual, cfg, arm, post)
                gripper_state[arm] = post
            except Exception as exc:
                last_err = f"设置夹爪 {post} 失败: {exc}"
                logger.error("[%s] %s", arm, last_err)
                return False, last_err
            if grip_dwell > 0:
                time.sleep(grip_dwell)

            try:
                load = float(dual.arms[arm].read_load())
                pos = float(dual.arms[arm].read_gripper())
                logger.info("[%s] 第 %d 步后 夹爪 pos=%.1f load=%.1f",
                            arm, index, pos, load)
            except Exception:
                load = 0.0

            if (verify_grasp and post == "closed"
                    and load_info.get(arm, {}).get("available")):
                thresh = float(load_info[arm]["thresh"])
                if load < thresh:
                    last_err = (f"合爪负载 {load:.1f} < 阈值 {thresh:.1f}，"
                                f"可能没夹住")
                    logger.warning("[%s] %s", arm, last_err)
                    if attempt < int(tries):
                        # 重试同一个步骤：回到 pre（通常是张开），再走同一 q。
                        try:
                            set_gripper_only(dual, cfg, arm, pre)
                            gripper_state[arm] = pre
                        except Exception:
                            pass
                        continue
                    return False, last_err

        return True, ""

    return False, last_err or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="严格按 14 步示教动作流循环执行（默认无限循环）")
    ap.add_argument("--config", default=None, help="config.json 路径")
    ap.add_argument("--flow", default=str(FLOW_PATH), help="taught_flow.json 路径")
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="运行分钟数；0=不限时")
    ap.add_argument("--cycles", type=int, default=0,
                    help="完整 14 步轮数；0=不限")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="全局速度倍率（0.05~2.0，首跑建议 0.5~0.7）")
    ap.add_argument("--tries", type=int, default=1,
                    help="每步失败后重复同一示教位的次数（默认 1=严格一遍）")
    ap.add_argument("--no-verify-grasp", action="store_true",
                    help="不检查合爪负载（只按位置和顺序执行）")
    ap.add_argument("--grip-dwell", type=float, default=0.2,
                    help="合爪/开爪后的额外停留秒数")
    ap.add_argument("--cycle-pause", type=float, default=0.1,
                    help="一轮结束后的额外停留秒数")
    ap.add_argument("--stall-restart", type=float, default=300.0,
                    help="连续多少秒没有成功轮次就退出等待重启；0=禁用")
    ap.add_argument("--csv", default=str(HERE / "logs" / "taught_flow.csv"),
                    help="CSV 日志路径")
    ap.add_argument("--no-csv", action="store_true", help="不写 CSV")
    ap.add_argument("--no-precondition", action="store_true",
                    help="启动时不预置到安全位；要求两臂已经在安全位")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印 14 步，不连接硬件")
    ap.add_argument("--check", action="store_true",
                    help="只读校验位姿/指纹，不使能力矩")
    args = ap.parse_args()

    if not (0.05 <= args.speed <= 2.0):
        ap.error("--speed 必须在 0.05 ~ 2.0 之间")
    if args.minutes < 0 or args.cycles < 0 or args.tries < 1:
        ap.error("--minutes/--cycles 不能为负，--tries 至少为 1")

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="run_taught_flow")
    flow_path = Path(args.flow)
    flow = load_flow(flow_path)
    problems, warnings = validate_flow(flow, cfg, require_complete=True)

    if args.dry_run:
        print_plan(flow, cfg)
        for line in warnings:
            print(line)
        if problems:
            print("\n阻断问题：")
            for line in problems:
                print(line)
            print("\n动作流还不完整；先运行 python calibrate_taught_flow.py。",
                  file=sys.stderr)
            return 2
        return 0

    if args.check:
        return cmd_check(cfg, logger, flow_path)

    for line in warnings:
        logger.warning("%s", line)
    if problems:
        for line in problems:
            logger.error("%s", line)
        print("动作流不完整或不可用，已拒绝启动。", file=sys.stderr)
        return 2

    stop_event = threading.Event()

    def on_signal(signum, _frame):
        logger.warning("收到信号 %s，准备停止", signum)
        stop_event.set()
        raise C.StopRequested()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    dual: DualArm | None = None
    csv_log: CycleCSV | None = None
    ok_all = False
    try:
        dual = DualArm(cfg, logger, stop_event)
        firmware = read_firmware(dual.ports, C.ARMS, logger)
        missing = [a for a in C.ARMS if a not in firmware]
        if missing:
            for arm in missing:
                logger.error("[%s] 无法读取固件标定，拒绝运动", arm)
            print("无法确认示教点属于当前电机标定，已拒绝运动。",
                  file=sys.stderr)
            return 2
        fp_problems, fp_warnings = flow_fingerprint_report(flow, firmware)
        for line in fp_warnings:
            logger.warning("%s", line)
        if fp_problems:
            for line in fp_problems:
                logger.error("%s", line)
            print("示教点与当前电机标定不一致，已拒绝运动；"
                  "请重新运行 calibrate_taught_flow.py。", file=sys.stderr)
            return 2

        dual.connect()
        mover = Mover(cfg, dual, logger, stop_event)
        mover.base_speed = float(args.speed)

        # 夹爪初始张开，并读一次空载以建立合爪判定阈值。
        load_info = {}
        for arm in C.ARMS:
            try:
                open_g = gripper_values(cfg, arm)[0]
                set_gripper_only(dual, cfg, arm, "open",
                                 settle=float(cfg["motion"]["gripper_open_settle_s"]))
                air_load = float(dual.arms[arm].read_load())
                min_add = float(cfg["grasp"]["load_min"])
                available = 0.5 < air_load < 300.0
                load_info[arm] = {
                    "air": air_load,
                    "thresh": air_load + min_add,
                    "available": available,
                }
                logger.info("[%s] 空载 load=%.1f 合爪阈值=%.1f 负载可用=%s",
                            arm, air_load, load_info[arm]["thresh"], available)
            except Exception as exc:
                logger.warning("[%s] 读空载失败，关闭合爪负载检查: %s", arm, exc)
                load_info[arm] = {"available": False, "thresh": 0.0}

        verify_grasp = not args.no_verify_grasp

        # 启动时先把两臂摆到各自安全位，避免右臂在左臂动作时停在未知位置。
        # 这只是把两臂放到示教安全位，不产生新位置；--no-precondition 可跳过。
        if not args.no_precondition:
            for safe_index in (1, 8):
                step = flow["steps"][safe_index - 1]
                arm = step["arm"]
                open_g = gripper_values(cfg, arm)[0]
                try:
                    mover.mv(arm, [np.asarray(step["q"], dtype=float)],
                             gripper=open_g, label=f"启动预置 {step['label']}",
                             speed=0.5)
                except Exception as exc:
                    logger.warning("[%s] 启动预置安全位失败: %s", arm, exc)

        gripper_state = {"left": "open", "right": "open"}
        if not args.no_csv:
            csv_log = CycleCSV(Path(args.csv))

        t_start = time.perf_counter()
        deadline = (t_start + float(args.minutes) * 60.0
                    if args.minutes > 0 else None)
        max_cycles = int(args.cycles)
        completed = 0
        completed_ok = 0
        last_ok_time = time.perf_counter()
        last_cycle_time = time.perf_counter()

        while True:
            if (max_cycles > 0 and completed >= max_cycles):
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break

            # 只在轮次边界做温度/总线处理，不改变动作流姿态。
            if not check_temp(dual, cfg, logger, stop_event, mover):
                return 3
            ensure_bus(dual, cfg, logger)
            if (args.stall_restart > 0
                    and time.perf_counter() - last_ok_time > args.stall_restart):
                logger.error("连续 %.0fs 没有成功轮次，退出等待 systemd 重启",
                             args.stall_restart)
                return 3

            cycle_ok = True
            cycle_err = ""
            for step in flow["steps"]:
                ok, err = run_step(
                    mover, dual, cfg, logger, step, gripper_state, load_info,
                    tries=args.tries, verify_grasp=verify_grasp,
                    grip_dwell=args.grip_dwell, stop_event=stop_event)
                if not ok:
                    cycle_ok = False
                    cycle_err = f"第{step['index']}步: {err}"
                    logger.warning("第 %d 步「%s」未成功: %s",
                                   step["index"], step["label"], err)

            completed += 1
            duration = time.perf_counter() - last_cycle_time
            last_cycle_time = time.perf_counter()
            if cycle_ok:
                completed_ok += 1
                last_ok_time = time.perf_counter()
            logger.info("# 第 %d 轮%s（%.1fs，成功 %d/%d）",
                        completed, "完成" if cycle_ok else "有失败步骤",
                        duration, completed_ok, completed)
            if csv_log:
                csv_log.write(completed, cycle_ok, duration, cycle_err)

            if args.cycle_pause > 0:
                time.sleep(args.cycle_pause)

        logger.info("动作流结束：完成 %d 轮（成功 %d 轮）。",
                    completed, completed_ok)
        ok_all = True
        return 0
    except C.StopRequested:
        logger.warning("收到停止请求")
        return 130
    except Exception as exc:
        logger.error("运行失败: %s: %s", type(exc).__name__, exc)
        logger.debug(traceback.format_exc())
        return 1
    finally:
        if csv_log is not None:
            csv_log.close()
        shutdown(dual, logger)
        logger.info("退出%s", "正常" if ok_all else "（未完成）")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断，退出清理会开爪并卸力。")
        sys.exit(130)
