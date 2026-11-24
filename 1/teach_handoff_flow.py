#!/usr/bin/env python
"""teach_handoff_flow.py — 10 步手递动作流示教（无摄像头）。

预设 10 步（每条臂）：
  1 home            夹爪 open
  2 carry           open
  3 table_approach  open
  4 table_hover     open
  5 table_pick      open  -> 到位后合爪
  6 table_hover     closed
  7 carry           closed
  8 place_right/left closed -> 到位后开爪
  9 carry           open
 10 home            open

示教只改每一步的 5 个关节角度；动作顺序、夹爪动作、停留时间都是预设的。
运行 run_handoff_flow.py 时，这 10 步的 q 和夹爪目标会原样发送。

用法（conda activate hupy3.12）：
  cd /home/zhoulin/hu/square2box/1

  python teach_handoff_flow.py --seed           # 用当前 config 生成两份预设 flow
  python teach_handoff_flow.py                  # 依次微调两条臂的 10 步
  python teach_handoff_flow.py --arm left       # 只示教左臂
  python teach_handoff_flow.py --step 8         # 只重教第 8 步（对方前方）
  python teach_handoff_flow.py --check          # 只读校验 flow 和指纹

操作：程序先移动到该步的参考位置，然后卸力；用手微调关节，回车记录。
  t 加力保持/卸力；s 保留当前参考值；q 退出。
"""

from __future__ import annotations

import argparse
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

import armcore as C
from armdriver import DualArm
from flow_core import (
    FLOW_TEMPLATE,
    N_STEPS,
    build_flow,
    flow_path,
    load_flow,
    print_flow,
    save_flow,
    validate_flow,
)
from handoff_core import (
    CROSS_KEY,
    fingerprint_report,
    pose_margins,
    read_firmware,
    read_ports,
    shutdown,
    validate_config,
)

HERE = Path(__file__).resolve().parent


def ask(prompt: str, default: str = "") -> str:
    try:
        return input(prompt).strip() or default
    except (EOFError, KeyboardInterrupt):
        return "q"


def set_torque_on(arm, logger) -> bool:
    try:
        bus = arm.bus.bus
        if bus is None:
            raise RuntimeError("臂未连接")
        names = arm.bus.joint_names + [arm.bus.gripper_name]
        pos = bus.sync_read("Present_Position", names)
        bus.sync_write("Goal_Position", {n: float(pos[n]) for n in names})
        arm.bus.enable_torque()
        return True
    except Exception as exc:
        logger.warning("加力失败: %s", exc)
        return False


def pose_fingerprints(cfg: dict, arm: str,
                      include_cross: bool = True) -> set[str]:
    """该臂点位的短指纹集合。"""
    keys = list(C.POSE_KEYS) + ([CROSS_KEY[arm]] if include_cross else [])
    cal = cfg["arms"][arm].get("pose_calib") or {}
    out = set()
    for key in keys:
        if (cfg["arms"][arm].get("poses") or {}).get(key) is None:
            continue
        fps = cal.get(key)
        if fps:
            out.add(str(fps))
    return out


def cmd_seed(cfg: dict, flow_dir: Path) -> int:
    problems, warnings = validate_config(cfg, require_cross=True)
    for line in warnings:
        print(line)
    if problems:
        print("\n阻断问题：")
        for line in problems:
            print(line)
        print("\n无法生成预设 flow；请先完成上面的标定。")
        return 2

    for arm in C.ARMS:
        mixed = pose_fingerprints(cfg, arm, include_cross=True)
        if len(mixed) > 1:
            print(f"[{arm}] 该臂的点位混用了多套电机标定指纹: {sorted(mixed)}")
            print("  这种配置不能生成 flow；请完整重标该臂。")
            return 2
        try:
            flow = build_flow(cfg, arm, require_cross=True)
        except SystemExit as exc:
            print(f"[{arm}] {exc}")
            return 2
        path = flow_path(flow_dir, arm)
        save_flow(path, flow)
        print(f"已生成 {path}")

    print("\n预设 10 步 flow 已生成。可查看：")
    for arm in C.ARMS:
        print_flow(load_flow(flow_path(flow_dir, arm)))
    print("\n下一步：")
    print("  python teach_handoff_flow.py              # 逐点微调")
    print("  python teach_handoff_flow.py --step 8     # 只调对方前方交接点")
    print("  ./run_handoff_flow.sh --check")
    return 0


def cmd_check(cfg: dict, flow_dir: Path, logger) -> int:
    print("=" * 72)
    print("只读校验（--check：不使能力矩，不发送运动指令）")
    print("=" * 72)
    problems = []
    warnings = []
    flows = {}
    for arm in C.ARMS:
        path = flow_path(flow_dir, arm)
        if not path.is_file():
            problems.append(f"[{arm}] 缺少 {path}；先运行: python teach_handoff_flow.py --seed")
            continue
        flows[arm] = load_flow(path)

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


def teach_arm(cfg: dict, dual: DualArm, arm: str, flow: dict,
              flow_dir: Path, step_filter: int | None,
              overwrite: bool, logger) -> bool:
    a = dual.arms[arm]
    open_g = float(cfg["arms"][arm]["gripper_open"])
    close_g = float(cfg["arms"][arm]["gripper_closed"])
    steps = flow["steps"]
    print(f"\n[{arm}] 预设 10 步：")
    for s in steps:
        after = s.get("grip_after")
        after_s = "" if after is None else f" -> {float(after):.1f}"
        print(f"  {s['index']:>2}. {s['label']:<16} 夹爪 {float(s['gripper']):.1f}{after_s}")
    print("  示教只改每一步的关节角度；动作顺序和夹爪动作是预设的。")

    for step in steps:
        idx = int(step["index"])
        if step_filter is not None and idx != step_filter:
            continue
        label = step.get("label", f"step{idx}")
        ref_q = np.asarray(step["q"], dtype=float)
        print("\n" + "-" * 70)
        print(f"[{arm}] 第 {idx}/{N_STEPS} 步：{label}  "
              f"（移动夹爪 {float(step['gripper']):.1f}）")
        if not overwrite and (step.get("source") == "taught" or step_filter is None):
            ans = ask("  回车重新示教 / s 保留当前值: ")
            if ans.lower() == "s":
                print("  保留。")
                continue

        # 先移动到参考位，方便在已标定姿态附近微调。
        try:
            a.move([ref_q], gripper=float(step["gripper"]),
                   label=f"{arm} 第 {idx} 步参考位",
                   speed_scale=0.5, timeout=10.0)
        except Exception as exc:
            logger.warning("[%s] 去第 %d 步参考位未完成: %s", arm, idx, exc)
        try:
            a.set_gripper(float(step["gripper"]),
                          settle=float(cfg["motion"]["gripper_close_settle_s"]))
        except Exception:
            pass

        print(f"  接下来会卸力 {arm} 臂，请先托住它。")
        ask("  准备好按回车（q 退出）")
        a.bus.disable_torque()
        torque_on = False
        print("  微调 5 个关节；回车记录 / t 加力保持或卸力 / s 保留参考值 / q 退出")

        while True:
            ans = ask("  > ")
            if ans.lower() == "q":
                raise SystemExit(0)
            if ans.lower() == "s":
                print("  保留参考值。")
                if not torque_on:
                    set_torque_on(a, logger)
                break
            if ans.lower() == "t":
                if torque_on:
                    a.bus.disable_torque()
                    torque_on = False
                    print("  已卸力，请托住。")
                elif set_torque_on(a, logger):
                    torque_on = True
                    print("  已加力，可以松手；按 t 再卸力。")
                continue
            if ans:
                print("  只支持 回车 / t / s / q。")
                continue

            q = np.asarray(a.read_joints(), dtype=float)
            bad = C.saturated_joints(q)
            margins = pose_margins(q)
            tight = [f"{n}({m:.1f})" for n, m in margins if m < 1.0]
            near = [f"{n}({m:.1f})" for n, m in margins if 1.0 <= m < 2.0]
            if bad or tight:
                print(f"  ! 当前位姿不可用：截断={bad or '无'}，贴边={tight or '无'}")
                print("  请微调后再按回车（尚未保存）。")
                continue
            if near:
                print(f"  · 提醒：这些关节离边界很近: {', '.join(near)}")
            if not torque_on:
                if not set_torque_on(a, logger):
                    print("  ! 加力失败，请托住机械臂。")
                else:
                    torque_on = True
            step["q"] = [round(float(v), 4) for v in q]
            step["source"] = "taught"
            flow["created"] = datetime.now().isoformat(timespec="seconds")
            cal = getattr(a.bus.bus, "calibration", None)
            if cal:
                flow["fingerprint"] = C.calib_fingerprint_short(cal)
            save_flow(flow_path(flow_dir, arm), flow)
            print(f"  已保存第 {idx} 步: {C.fmt_q(q)}")
            break

    # 流程结束再写一次，确保 fingerprint/created 是当前值。
    cal = getattr(a.bus.bus, "calibration", None)
    if cal:
        flow["fingerprint"] = C.calib_fingerprint_short(cal)
    flow["created"] = datetime.now().isoformat(timespec="seconds")
    save_flow(flow_path(flow_dir, arm), flow)
    try:
        a.move(["home"], gripper=open_g, label=f"{arm} 示教后回 home",
               speed_scale=0.6, timeout=10.0)
    except Exception as exc:
        logger.warning("[%s] 示教后回 home 未完成: %s", arm, exc)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="10 步手递动作流示教")
    ap.add_argument("--arm", choices=("left", "right", "both"), default="both")
    ap.add_argument("--step", type=int, default=0,
                    help="只重教第 N 步（1~10）；0=全部")
    ap.add_argument("--overwrite", action="store_true",
                    help="不询问，直接重教选中的步骤")
    ap.add_argument("--seed", action="store_true",
                    help="用当前 config 生成两份预设 flow，不连硬件")
    ap.add_argument("--check", action="store_true",
                    help="只读校验 flow/指纹，不动电机")
    ap.add_argument("--config", default=None)
    ap.add_argument("--flow-dir", default=str(HERE / "flows"))
    args = ap.parse_args()

    if args.step and not (1 <= args.step <= N_STEPS):
        ap.error(f"--step 必须在 1~{N_STEPS}")
    if args.step and args.arm == "both":
        ap.error("--step 需要配合 --arm left 或 --arm right")

    cfg_path = Path(args.config) if args.config else C.CONFIG_PATH
    flow_dir = Path(args.flow_dir)
    cfg = C.load_config(cfg_path)
    logger = C.setup_logging(cfg, name="teach_handoff_flow")

    if args.seed:
        return cmd_seed(cfg, flow_dir)
    if args.check:
        return cmd_check(cfg, flow_dir, logger)

    problems, warnings = validate_config(cfg, require_cross=False)
    for line in warnings:
        logger.warning("%s", line)
    if problems:
        for line in problems:
            logger.error("%s", line)
        print("基础点位不完整，先重跑 calibrate2arm.py。", file=sys.stderr)
        return 2

    arms = ["left", "right"] if args.arm == "both" else [args.arm]
    dual: DualArm | None = None
    ok = False
    try:
        dual = DualArm(cfg, logger, threading.Event())
        firmware = read_firmware(dual.ports, C.ARMS, logger)
        missing = [a for a in C.ARMS if a not in firmware]
        if missing:
            print(f"无法读取固件标定: {missing}", file=sys.stderr)
            return 2
        fp_problems, fp_warnings = fingerprint_report(
            cfg, firmware, include_cross=False)
        for line in fp_warnings:
            logger.warning("%s", line)
        if fp_problems:
            for line in fp_problems:
                logger.error("%s", line)
            print("当前 10 个点位不是在当前电机标定下录的；请先完整重跑 "
                  "calibrate2arm.py。", file=sys.stderr)
            return 2
        for arm in C.ARMS:
            mixed = pose_fingerprints(cfg, arm, include_cross=False)
            if len(mixed) > 1:
                print(f"[{arm}] 基础点位混用了多套标定指纹: {sorted(mixed)}；"
                      f"请先完整重标该臂。", file=sys.stderr)
                return 2
            cross_fp = (cfg["arms"][arm].get("pose_calib") or {}).get(CROSS_KEY[arm])
            base_fps = pose_fingerprints(cfg, arm, include_cross=False)
            if cross_fp and base_fps and str(cross_fp) not in base_fps:
                logger.warning("[%s] 交叉位 %s 是旧标定指纹；"
                               "示教时请重录第 8 步", arm, CROSS_KEY[arm])

        dual.connect()
        flows = {}
        for arm in arms:
            path = flow_path(flow_dir, arm)
            if path.is_file():
                flow = load_flow(path)
                if len(flow.get("steps") or []) != N_STEPS:
                    logger.warning("[%s] 现有 flow 不是 10 步，改用 config 预设", arm)
                    flow = build_flow(cfg, arm, require_cross=False)
            else:
                flow = build_flow(cfg, arm, require_cross=False)
                logger.info("[%s] 生成预设 flow（来自 config）", arm)
            flows[arm] = flow
            teach_arm(cfg, dual, arm, flow, flow_dir, args.step,
                      args.overwrite, logger)

        for arm in arms:
            print_flow(flows[arm])
        print("\n下一步：")
        print("  ./run_handoff_flow.sh --check")
        print("  ./run_handoff_flow.sh --dry-run")
        print("  ./run_handoff_flow.sh --minutes 5 --speed 0.7")
        ok = True
        return 0
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\n已中断，退出会开爪并卸力。")
        return 130
    except Exception as exc:
        logger.error("示教失败: %s: %s", type(exc).__name__, exc)
        return 1
    finally:
        shutdown(dual, logger)
        logger.info("退出%s", "正常" if ok else "（未完成）")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)
