#!/usr/bin/env python
"""dualarm_handoff_loop.py — 双臂循环手递夹放（无摄像头）。

一轮动作：
  1. 左臂从 left_front（左臂自己的 table_pick）抓起方块
  2. 左臂放到 right_front（右臂 table_pick 的位置，用 left.place_right）
  3. 右臂从 right_front（自己的 table_pick）抓起
  4. 右臂放到 left_front（左臂 table_pick 的位置，用 right.place_left）
  → 方块回到 left_front，计 1 轮，继续循环

两个“前方交接点”是两条臂各自的 table_pick 物理位置。每条臂到对方
交接点的关节位姿由 calibrate_handoff.py 一次性示教（place_right /
place_left）。不同臂的关节值不能互换，所以这两个交叉位必须实测。

可靠性设计：
  * 运行目标只使用示教位姿：home / carry / table_approach / table_hover /
    table_pick / place_right / place_left，不叠加释放抬高、pan/深度偏移
    或反推出来的 hover。
  * 空夹基线 + 位置/负载双重验证；抓空时重复同一个示教点（不换 pan/深度）。
  * 一抓一放失败后按“方块在哪个交接点”继续，而不是整轮重来；
    方块位置未知时用两条臂在左右两个交接点做四点搜索。
  * 温度降速/冷却、总线重连、卡死退出（systemd 可 Restart=always）。
  * 默认无限循环，--minutes/--cycles 可限时/限量。

用法（conda activate hupy3.12）：
  cd /home/zhoulin/hu/square2box/1
  python calibrate_handoff.py            # 首次：示教两个交叉位
  ./run_handoff.sh --check               # 只读校验
  ./run_handoff.sh --dry-run             # 查看动作流
  ./run_handoff.sh --minutes 5           # 连续跑 5 分钟
  ./run_handoff.sh                       # 无限循环，直到 Ctrl+C

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

import armcore as C
from armdriver import DualArm
from handoff_core import (
    CROSS_KEY,
    OWN_KEY,
    HandoffRunner,
    HandoffState,
    fingerprint_report,
    read_firmware,
    read_ports,
    shutdown,
    validate_config,
)

HERE = Path(__file__).resolve().parent


class CycleCSV:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["time", "cycle", "status", "arm", "front",
                             "duration_s", "error"])

    def write(self, cycle: int, status: str, arm: str = "",
              front: str = "", duration: float = 0.0, error: str = "") -> None:
        self.w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), cycle, status,
                         arm, front, f"{duration:.1f}", str(error)[:300]])
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


def print_plan(cfg: dict, overlap: bool) -> None:
    print("=" * 72)
    print("双臂循环手递（无摄像头）动作流")
    print("=" * 72)
    print("初始：方块在 left_front（左臂自己的 table_pick）")
    print("所有箭头上的目标都是 config.json 里的示教位姿，运行时不会修改。")
    print()
    print("  1. left : home -> carry -> table_approach -> table_hover"
          " -> table_pick    [抓取]")
    print(f"  2. left : table_pick -> table_hover -> carry -> "
          f"{CROSS_KEY['left']}    [释放]")
    print("  过渡    : left 回 carry -> home；right 预位 home -> carry"
          " -> table_approach")
    print("  3. right: table_approach -> table_hover -> table_pick"
          "    [抓取]")
    print(f"  4. right: table_pick -> table_hover -> carry -> "
          f"{CROSS_KEY['right']}    [释放]")
    print("  过渡    : right 回 carry -> home；left 预位 home -> carry"
          " -> table_approach")
    print("  → 方块回到 left_front，计 1 轮；随后重复")
    print()
    print("失败恢复：")
    print("  - 某个点抓空：同一交接点先换另一条臂（它也有一组位姿）重试；")
    print("  - 两边都抓不到：判定方块位置未知，按 left/right 两个交接点做四点搜索；")
    print("  - 放置下探失败但夹爪仍夹住：下一步原地重放，不整轮重启。")
    print(f"  - 过渡方式：{'并行预位（减少停顿）' if overlap else '先回 home 再预位'}")
    print()
    print("注意：交叉位必须由 calibrate_handoff.py 一次性示教；")
    print("      两条臂的关节值不能互换，不能拿对方的 table_pick 直接发给自己。")
    missing = [a for a in ("left", "right")
               if (cfg["arms"][a].get("poses") or {}).get(CROSS_KEY[a]) is None]
    if missing:
        print()
        print("当前缺少交叉位：")
        for arm in missing:
            print(f"  [{arm}] 需要 {CROSS_KEY[arm]} -> "
                  f"python calibrate_handoff.py --arm {arm}")


def print_rows(title: str, rows: list[str]) -> None:
    if not rows:
        return
    print(f"\n{title}")
    for line in rows:
        print(line)


def cmd_check(cfg: dict, logger) -> int:
    print("=" * 72)
    print("只读校验（--check：不使能力矩，不发送运动指令）")
    print("=" * 72)

    problems, warnings = validate_config(cfg, require_cross=True)
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
        fp_problems, fp_warnings = fingerprint_report(
            cfg, firmware, include_cross=True)
        problems.extend(fp_problems)
        warnings.extend(fp_warnings)

    print_rows("警告（不阻断）", warnings)
    if problems:
        print_rows("阻断问题", problems)
        print(f"\n校验失败：{len(problems)} 个阻断问题。")
        return 1
    print("\n校验通过：位姿和交叉位齐全，未发现阻断问题。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="双臂循环手递夹放（无摄像头，默认无限循环）")
    ap.add_argument("--config", default=None,
                    help="config.json 路径（默认 1/config.json）")
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="运行分钟数；0=不限时（默认无限循环）")
    ap.add_argument("--cycles", type=int, default=0,
                    help="完整轮数（方块回到 left_front）；0=不限轮")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="全局速度倍率（0.05~2.0，首跑建议 0.5~0.7）")
    ap.add_argument("--pick-tries", type=int, default=3,
                    help="每个抓取点重复同一示教位的次数（默认 3，不做偏移）")
    ap.add_argument("--no-overlap", action="store_true",
                    help="放置臂先回 home，再让下一条臂预位（默认并行预位）")
    ap.add_argument("--cycle-pause", type=float, default=0.1,
                    help="一轮结束后的额外停留秒数")
    ap.add_argument("--stall-restart", type=float, default=300.0,
                    help="连续多少秒无进展就退出等待重启；0=禁用")
    ap.add_argument("--csv", default=str(HERE / "logs" / "handoff.csv"),
                    help="CSV 日志路径")
    ap.add_argument("--no-csv", action="store_true", help="不写 CSV")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印动作流，不连接硬件")
    ap.add_argument("--check", action="store_true",
                    help="只读校验位姿/指纹，不使能力矩")
    ap.add_argument("--home-only", action="store_true",
                    help="两臂开爪回 home 后退出")
    args = ap.parse_args()

    if not (0.05 <= args.speed <= 2.0):
        ap.error("--speed 必须在 0.05 ~ 2.0 之间")
    if args.minutes < 0 or args.cycles < 0 or args.pick_tries < 1:
        ap.error("--minutes/--cycles 不能为负，--pick-tries 至少为 1")

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="handoff_loop")
    require_cross = not args.home_only

    problems, warnings = validate_config(cfg, require_cross=require_cross)

    if args.dry_run:
        print_plan(cfg, not args.no_overlap)
        print_rows("配置提示（不阻断运行）", warnings)
        if problems:
            print_rows("阻断问题", problems)
            print("\n当前配置还不能运行；请先完成上面的标定。", file=sys.stderr)
            return 2
        return 0

    if args.check:
        return cmd_check(cfg, logger)

    for line in warnings:
        logger.warning("%s", line)
    if problems:
        for line in problems:
            logger.error("%s", line)
        print("标定配置不完整或不可用，已拒绝启动。", file=sys.stderr)
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
    ok = False
    try:
        dual = DualArm(cfg, logger, stop_event)
        firmware = read_firmware(dual.ports, C.ARMS, logger)
        fp_problems, fp_warnings = fingerprint_report(
            cfg, firmware, include_cross=require_cross)
        for line in fp_warnings:
            logger.warning("%s", line)
        missing_fw = [a for a in C.ARMS if a not in firmware]
        if missing_fw:
            for arm in missing_fw:
                logger.error("[%s] 无法读取固件标定，拒绝运动", arm)
            print("无法确认点位属于当前电机标定，已拒绝运动。",
                  file=sys.stderr)
            return 2
        if fp_problems:
            for line in fp_problems:
                logger.error("%s", line)
            print("点位与当前电机标定不一致，已拒绝运动；"
                  "请先重跑 calibrate2arm.py / calibrate_handoff.py 并同步。",
                  file=sys.stderr)
            return 2

        dual.connect()

        runner = HandoffRunner(
            cfg, dual, logger, stop_event,
            pick_tries=args.pick_tries,
            overlap=not args.no_overlap)
        runner.mover.base_speed = float(args.speed)

        if args.home_only:
            runner.home_both()
            ok = True
            logger.info("已回 home。")
            return 0

        runner.grasp.measure_air_closed()
        runner.home_both()
        state = HandoffState(runner, logger, pick_tries=args.pick_tries)

        if not args.no_csv:
            csv_log = CycleCSV(Path(args.csv))

        t_start = time.perf_counter()
        deadline = (t_start + float(args.minutes) * 60.0
                    if args.minutes > 0 else None)
        max_cycles = int(args.cycles)
        deadline_grace = 60.0
        completed = 0
        steps = 0
        last_progress = time.perf_counter()
        last_cycle = time.perf_counter()

        while True:
            now = time.perf_counter()
            if (deadline is not None and now >= deadline
                    and state.cube_hand is None):
                if state.cube_front == "left" or now >= deadline + deadline_grace:
                    break
            if (max_cycles > 0 and completed >= max_cycles
                    and state.cube_hand is None and state.cube_front == "left"):
                break

            runner.check_temp(allow_cool=(state.cube_hand is None))
            if state.cube_hand is None:
                runner.ensure_bus()

            result = state.step()
            steps += 1
            status = result["status"]
            arm = result.get("arm", "")
            front = result.get("front", "")
            if status in ("picked", "placed", "recovered"):
                last_progress = time.perf_counter()

            if result.get("cycle"):
                completed += 1
                duration = time.perf_counter() - last_cycle
                last_cycle = time.perf_counter()
                logger.info("# 第 %d 轮完成（用时 %.1fs，累计 %d 轮，%d 步）",
                            completed, duration, completed, steps)
                if csv_log:
                    csv_log.write(completed, "cycle", arm, front, duration)

            if status in ("pick_failed", "lost", "error", "not_found"):
                logger.warning("[状态] %s%s", status,
                               f": {result.get('error')}" if result.get("error") else "")
                if csv_log:
                    csv_log.write(completed, status, arm, front,
                                  error=result.get("error", ""))

            if status == "not_found":
                # HandoffState 已在四点搜索失败后把两臂送回 home。
                if args.cycle_pause > 0:
                    time.sleep(args.cycle_pause)
                if (args.stall_restart > 0
                        and time.perf_counter() - last_progress > args.stall_restart):
                    logger.error("连续 %.0fs 无进展，退出等待 systemd 重启",
                                 args.stall_restart)
                    return 3
            elif result.get("cycle") and args.cycle_pause > 0:
                time.sleep(args.cycle_pause)

            if (max_cycles > 0 and completed >= max_cycles
                    and state.cube_hand is None and state.cube_front == "left"):
                break
            if (deadline is not None and state.cube_hand is None
                    and time.perf_counter() >= deadline + deadline_grace):
                break

        logger.info("循环结束：完成 %d 轮，共 %d 步。", completed, steps)
        runner.home_both()
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
