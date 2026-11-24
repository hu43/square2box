#!/usr/bin/env python
"""run_handoff_flow.py — 按示教好的 10 步 flow 原样运行（无摄像头）。

每个轮次：
  左臂 10 步 flow（home -> ... -> place_right -> ... -> home）
  右臂 10 步 flow（home -> ... -> place_left  -> ... -> home）
  重复。

运行只读 flow JSON 里的 q 和夹爪目标，不叠加任何偏移/抬高/派生位姿。
flow 由 teach_handoff_flow.py --seed 生成，也可用交互模式逐点微调。

用法：
  cd /home/zhoulin/hu/square2box/1
  python teach_handoff_flow.py --seed
  ./run_handoff_flow.sh --check
  ./run_handoff_flow.sh --dry-run
  ./run_handoff_flow.sh --minutes 5 --speed 0.7
  ./run_handoff_flow.sh                     # 无限循环
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

import armcore as C
from armdriver import DualArm
from flow_core import (
    FlowRunner,
    flow_path,
    load_flow,
    print_flow,
    validate_flow,
)
from handoff_core import read_firmware, read_ports, shutdown

HERE = Path(__file__).resolve().parent


class FlowCSV:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["time", "cycle", "arm", "status",
                             "duration_s", "error"])

    def write(self, cycle: int, arm: str, status: str,
              duration: float = 0.0, error: str = "") -> None:
        self.w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), cycle, arm,
                         status, f"{duration:.1f}", str(error)[:300]])
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


def load_flows(flow_dir: Path) -> tuple[dict, list[str]]:
    flows = {}
    problems = []
    for arm in C.ARMS:
        path = flow_path(flow_dir, arm)
        if not path.is_file():
            problems.append(
                f"[{arm}] 缺少 {path}\n"
                f"  请先生成预设: python teach_handoff_flow.py --seed")
            continue
        try:
            flows[arm] = load_flow(path)
        except SystemExit as exc:
            problems.append(str(exc))
    return flows, problems


def cmd_check(cfg: dict, flow_dir: Path, logger) -> int:
    print("=" * 72)
    print("只读校验（--check：不使能力矩，不发送运动指令）")
    print("=" * 72)
    flows, problems = load_flows(flow_dir)
    warnings = []
    firmware = {}
    try:
        ports = read_ports(cfg, logger)
    except Exception as exc:
        print(f"\n串口解析失败，跳过固件指纹检查: {exc}")
        ports = None
    if ports:
        firmware = read_firmware(ports, C.ARMS, logger)
        for arm in C.ARMS:
            if arm not in firmware:
                warnings.append(f"[{arm}] 固件标定读取失败，跳过该臂指纹检查")

    for arm, flow in flows.items():
        p, w = validate_flow(flow, arm, firmware.get(arm))
        problems.extend(p)
        warnings.extend(w)
        print_flow(flow)

    for line in warnings:
        print(line)
    if problems:
        print("\n阻断问题：")
        for line in problems:
            print(line)
        print(f"\n校验失败：{len(problems)} 个阻断问题。")
        return 1
    print("\n校验通过：两份 10 步 flow 都可用。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="按示教好的 10 步 flow 原样运行（默认无限循环）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--flow-dir", default=str(HERE / "flows"))
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="运行分钟数；0=不限时")
    ap.add_argument("--cycles", type=int, default=0,
                    help="完整轮次（左右 flow 都成功）；0=不限")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="全局速度倍率（0.05~2.0）")
    ap.add_argument("--flow-tries", type=int, default=2,
                    help="单个 flow 失败后，原样重试几次（不改变位置）")
    ap.add_argument("--max-failures", type=int, default=3,
                    help="连续失败多少个轮次后退出等待重启；0=一直重试")
    ap.add_argument("--cycle-pause", type=float, default=0.15)
    ap.add_argument("--csv", default=str(HERE / "logs" / "handoff_flow.csv"))
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印 flow，不连硬件")
    ap.add_argument("--check", action="store_true", help="只读校验 flow/指纹")
    ap.add_argument("--home-only", action="store_true", help="两臂回 home 后退出")
    args = ap.parse_args()

    if not (0.05 <= args.speed <= 2.0):
        ap.error("--speed 必须在 0.05 ~ 2.0 之间")
    if args.minutes < 0 or args.cycles < 0 or args.flow_tries < 1:
        ap.error("--minutes/--cycles 不能为负，--flow-tries 至少为 1")

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="handoff_flow")
    flow_dir = Path(args.flow_dir)

    flows, load_problems = load_flows(flow_dir)

    if args.dry_run:
        print("=" * 72)
        print("10 步示教动作流（--dry-run：只读 flow，不连接硬件）")
        print("=" * 72)
        for arm in C.ARMS:
            if arm in flows:
                print_flow(flows[arm])
        if load_problems:
            print("\n阻断问题：")
            for line in load_problems:
                print(line)
            print("\n先生成预设: python teach_handoff_flow.py --seed",
                  file=sys.stderr)
            return 2
        print("\n每个轮次：left flow -> right flow -> 重复。")
        print("指纹需要在 --check 或正式运行时校验。")
        return 0

    if args.check:
        return cmd_check(cfg, flow_dir, logger)

    if not args.home_only:
        if load_problems:
            for line in load_problems:
                print(line, file=sys.stderr)
            return 2
        for arm in C.ARMS:
            p, w = validate_flow(flows[arm], arm, firmware=None)
            for line in w:
                logger.warning("%s", line)
            if p:
                # 指纹要等读到固件后才能校验；这里先报结构问题。
                structural = [x for x in p if "指纹" not in x]
                for line in structural:
                    logger.error("%s", line)
                if structural:
                    return 2

    stop_event = threading.Event()

    def on_signal(signum, _frame):
        logger.warning("收到信号 %s，准备停止", signum)
        stop_event.set()
        raise C.StopRequested()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    dual: DualArm | None = None
    csv_log: FlowCSV | None = None
    ok = False
    try:
        dual = DualArm(cfg, logger, stop_event)

        if args.home_only:
            dual.connect()
            runner = FlowRunner(cfg, dual, logger, stop_event, speed=args.speed)
            runner.home_both(flows)
            ok = True
            logger.info("已回 home。")
            return 0

        firmware = read_firmware(dual.ports, C.ARMS, logger)
        missing = [a for a in C.ARMS if a not in firmware]
        if missing:
            for arm in missing:
                logger.error("[%s] 无法读取固件标定，拒绝运动", arm)
            print("无法确认 flow 属于当前电机标定，已拒绝运动。",
                  file=sys.stderr)
            return 2
        fp_problems = []
        for arm in C.ARMS:
            p, w = validate_flow(flows[arm], arm, firmware[arm])
            fp_problems.extend([x for x in p if "指纹" in x])
            for line in w:
                logger.warning("%s", line)
            for line in p:
                if "指纹" not in line:
                    logger.error("%s", line)
        if fp_problems:
            for line in fp_problems:
                logger.error("%s", line)
            print("flow 与当前电机标定不一致，已拒绝运动。", file=sys.stderr)
            return 2

        dual.connect()
        runner = FlowRunner(cfg, dual, logger, stop_event, speed=args.speed)

        if not args.no_csv:
            csv_log = FlowCSV(Path(args.csv))

        t_start = time.perf_counter()
        deadline = (t_start + float(args.minutes) * 60.0
                    if args.minutes > 0 else None)
        max_cycles = int(args.cycles)
        completed = 0
        failed = 0
        next_arm = "left"

        while True:
            # 只在轮次边界停止：left flow 完成后还要跑 right flow，
            # 让方块回到左臂前，下一次启动才接着得上。
            if (deadline is not None and next_arm == "left"
                    and time.perf_counter() >= deadline):
                break
            if (max_cycles > 0 and next_arm == "left"
                    and completed >= max_cycles):
                break
            runner.check_temp(flows)
            runner.ensure_bus()

            arm = next_arm
            flow_ok = False
            last_error = ""
            for attempt in range(1, int(args.flow_tries) + 1):
                t0 = time.perf_counter()
                try:
                    flow_ok = runner.run_flow(flows[arm])
                except C.StopRequested:
                    raise
                except Exception as exc:
                    flow_ok = False
                    last_error = f"{type(exc).__name__}: {exc}"
                dur = time.perf_counter() - t0
                if flow_ok:
                    logger.info("[%s] 10 步 flow 成功（第 %d 次尝试，%.1fs）",
                                arm, attempt, dur)
                    if csv_log:
                        csv_log.write(completed + 1, arm, "flow_ok", dur)
                    break
                logger.warning("[%s] 10 步 flow 失败（第 %d 次尝试，%.1fs）%s",
                               arm, attempt, dur,
                               f": {last_error}" if last_error else "")
                if csv_log:
                    csv_log.write(completed + 1, arm, "flow_fail",
                                  dur, last_error)
                runner.safe_home(flows)

            if not flow_ok:
                failed += 1
                if runner.has_holding():
                    logger.error("[%s] flow 失败且仍夹着方块；停止自动运动，"
                                 "请人工处理或由 systemd 重启", arm)
                    if csv_log:
                        csv_log.write(completed + 1, arm, "flow_fail_holding")
                    return 3
                runner.safe_home(flows)
                logger.error("[%s] flow 连续失败（%d 次）", arm, failed)
                if csv_log:
                    csv_log.write(completed + 1, arm, "flow_fail_cycle")
                if args.max_failures > 0 and failed >= args.max_failures:
                    logger.error("连续 %d 次 flow 失败，退出等待 systemd 重启",
                                 failed)
                    return 3
                # 重试同一条 flow（同一个 10 步序列，不改位置/顺序）。
                continue

            failed = 0
            if next_arm == "left":
                next_arm = "right"
            else:
                completed += 1
                next_arm = "left"
                logger.info("# 第 %d 轮完成：left flow + right flow", completed)
                if csv_log:
                    csv_log.write(completed, "both", "cycle_ok")
                if args.cycle_pause > 0:
                    time.sleep(args.cycle_pause)

        logger.info("循环结束：完成 %d 轮。", completed)
        runner.home_both(flows)
        ok = True
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
        logger.info("退出%s", "正常" if ok else "（未完成）")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断，退出清理会开爪并卸力。")
        sys.exit(130)
