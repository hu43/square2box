#!/usr/bin/env python
"""calibrate_taught_flow.py — 按 14 步预设动作流逐点示教。

预设顺序（左臂 7 步、右臂 7 步）：
  1 左臂安全位          8 右臂安全位
  2 左臂夹方块前        9 右臂夹方块前
  3 左臂夹方块         10 右臂夹方块
  4 左臂夹起在中转位   11 右臂夹起在中转位
  5 左臂放方块前       12 右臂放方块前
  6 左臂放方块         13 右臂放方块
  7 左臂回安全位       14 右臂回安全位

示教时：
  * 程序按预设 pre 状态自动张/合夹爪，并让夹爪电机一直保持力矩；
  * 5 个关节电机的力矩可以卸掉，方便用手摆臂（按 t 切换）；
  * 第 3/10 步记录后自动合爪，第 6/13 步记录后自动开爪；
  * 方块会随步骤流转：左臂放下 -> 右臂从同一位置抓起 -> 右臂放下。

用法（conda activate hupy3.12）：
  cd /home/zhoulin/hu/square2box/1
  python calibrate_taught_flow.py              # 依次示教 14 步
  python calibrate_taught_flow.py --arm left   # 只示教左臂 7 步
  python calibrate_taught_flow.py --step 4     # 只重录第 4 步
  python calibrate_taught_flow.py --check      # 只读校验

安全：示教臂卸力时请托住；退出会开爪并卸力。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

import armcore as C
from armdriver import DualArm
from handoff_core import pose_margins, read_firmware, read_ports, shutdown
from taught_flow_core import (
    FLOW_PATH,
    STEP_DEFS,
    flow_fingerprint_report,
    gripper_values,
    load_flow,
    normalize_flow,
    save_flow,
    set_gripper_only,
    set_joint_torque,
    step_gripper_text,
    validate_flow,
)

HERE = Path(__file__).resolve().parent


def ask(prompt: str, default: str = "") -> str:
    try:
        return input(prompt).strip() or default
    except (EOFError, KeyboardInterrupt):
        return "q"


STEP_HINT = {
    1: "摆到左臂休息位，避开桌面、方块和右臂；夹爪张开。",
    2: "摆到方块正上方，夹爪张开，便于竖直向下到方块。",
    3: "把张开的夹爪摆到刚好包住方块的位置；记录后会自动合爪。",
    4: "夹爪应已夹住方块；把左臂摆到中转高处，避免扫到桌面/右臂。",
    5: "夹爪保持夹住方块；摆到左臂放置点正上方。",
    6: "夹爪保持夹住方块；摆到方块要落下的位置；记录后会自动开爪。",
    7: "夹爪已张开；摆回左臂安全位。",
    8: "摆到右臂休息位，避开桌面、方块和左臂；夹爪张开。",
    9: "摆到方块正上方（方块在上一步左臂放下的位置），夹爪张开。",
    10: "把张开的夹爪摆到刚好包住同一个方块的位置；记录后会自动合爪。",
    11: "夹爪应已夹住方块；把右臂摆到中转高处。",
    12: "夹爪保持夹住方块；摆到右臂放置点正上方。",
    13: "夹爪保持夹住方块；摆到方块要落下的位置；记录后会自动开爪。",
    14: "夹爪已张开；摆回右臂安全位。下一轮左臂夹方块必须到同一个落点。",
}

CUBE_HINT = {
    1: "初始：把方块放在左臂夹方块(3) 的抓取位置（左臂前方）。",
    2: "方块应在左臂夹方块(3) 的位置。",
    3: "张开夹爪包住方块；记录后合爪，方块会留在爪里。",
    4: "夹着方块搬到中转位。",
    5: "夹着方块到左臂放置点上方。",
    6: "记录后开爪，方块会落在左臂放置点；这个点就是右臂夹方块(10) 的位置。",
    7: "方块留在左臂放置点，左臂回安全位。",
    8: "右臂回安全位；方块仍在左臂放置点。",
    9: "把右臂摆到方块正上方（方块在左臂放置点）。",
    10: "右臂张开夹爪包住同一个方块；记录后合爪。",
    11: "右臂夹着方块到中转位。",
    12: "夹着方块到右臂放置点上方。",
    13: "记录后开爪，方块会落在右臂放置点；下一轮左臂夹方块(3) 必须到同一个点。",
    14: "方块留在右臂放置点，右臂回安全位。",
}


def target_indices(args) -> list[int]:
    if args.step:
        if not (1 <= args.step <= len(STEP_DEFS)):
            raise SystemExit(f"--step 必须在 1..{len(STEP_DEFS)}")
        return [int(args.step)]
    if args.arm == "left":
        return list(range(1, 8))
    if args.arm == "right":
        return list(range(8, 15))
    return list(range(1, 15))


def print_header(index: int) -> dict:
    d = STEP_DEFS[index - 1]
    print("\n" + "=" * 72)
    print(f"[{index}/{len(STEP_DEFS)}] {d['label']}  "
          f"({d['arm']} / {d['name']})")
    print("=" * 72)
    print(f"  {STEP_HINT[index]}")
    print(f"  {CUBE_HINT[index]}")
    return d


def record_step(cfg: dict, dual: DualArm, flow: dict, flow_path: Path,
                index: int, firmware: dict, logger) -> bool:
    d = STEP_DEFS[index - 1]
    arm = d["arm"]
    a = dual.arms[arm]
    q = np.asarray(a.read_joints(), dtype=float)
    if q.shape != (C.NJ,) or not np.all(np.isfinite(q)):
        print(f"  ! 读关节失败或形状异常: {q}")
        return False

    bad = C.saturated_joints(q)
    tight = [f"{n}({m:.1f})" for n, m in pose_margins(q) if m < 1.0]
    near = [f"{n}({m:.1f})" for n, m in pose_margins(q) if 1.0 <= m < 2.0]
    if bad or tight:
        print(f"  ! 该点不可用：截断={bad or '无'}，贴边={tight or '无'}")
        print("  请把交接点往可达方向挪一点，或调整腕部姿态，再按回车重试。")
        return False
    if near:
        print(f"  · 提醒：这些关节离边界很近: {', '.join(near)}（仍可记录）")

    fw = firmware.get(arm)
    if fw is None:
        print(f"  ! 读不到 {arm} 臂固件标定，不能记录指纹。")
        return False

    fp = C.calib_fingerprint_short(fw)
    norm = normalize_flow(flow)
    flow.clear()
    flow.update(norm)
    flow["steps"][index - 1]["q"] = [round(float(v), 4) for v in q]
    flow["steps"][index - 1]["pose_calib"] = fp
    save_flow(flow, flow_path)

    print(f"\n  已保存第 {index} 步「{d['label']}」= {C.fmt_q(q)}")
    print(f"  指纹 {fp}  ->  {flow_path}")
    return True


def apply_post(cfg: dict, dual: DualArm, index: int) -> None:
    d = STEP_DEFS[index - 1]
    post = d["post"]
    if post is None:
        return
    arm = d["arm"]
    value = set_gripper_only(dual, cfg, arm, post)
    print(f"  {d['label']}：夹爪已{'张开' if post == 'open' else '合上'}"
          f"（目标 {value:.1f}）")
    try:
        load = dual.arms[arm].read_load()
        pos = dual.arms[arm].read_gripper()
        print(f"  夹爪读数 pos={pos:.1f} load={load:.1f}")
        if post == "closed" and load < float(cfg["grasp"]["load_min"]):
            print("  · 提醒：合爪负载偏低，方块可能没夹住；下一步搬运前请确认。")
    except Exception:
        pass


def park_arms(dual: DualArm, cfg: dict, firmware: dict, logger) -> None:
    for arm in C.ARMS:
        home = (cfg["arms"][arm].get("poses") or {}).get("home")
        recorded = (cfg["arms"][arm].get("pose_calib") or {}).get("home")
        fw = firmware.get(arm)
        now = C.calib_fingerprint_short(fw) if fw else None
        if home is None or now is None or recorded != now:
            logger.warning("[%s] 标准 home 缺失或属于旧电机标定，跳过预停", arm)
            continue
        try:
            open_g = gripper_values(cfg, arm)[0]
            dual.arms[arm].move(["home"], gripper=open_g,
                                label=f"{arm} 预停 home",
                                speed_scale=0.6, timeout=10.0)
        except Exception as exc:
            logger.warning("[%s] 预停 home 未完成: %s", arm, exc)


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
    print("\n校验通过：14 步动作齐全，且都属于当前电机标定。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="14 步预设动作流示教")
    ap.add_argument("--arm", choices=("left", "right", "both"), default="both",
                    help="示教哪条臂（默认 14 步全部）")
    ap.add_argument("--step", type=int, default=0,
                    help="只重录某一步（1..14）")
    ap.add_argument("--flow", default=str(FLOW_PATH), help="taught_flow.json 路径")
    ap.add_argument("--config", default=None, help="config.json 路径")
    ap.add_argument("--check", action="store_true", help="只读校验，不动电机")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else C.CONFIG_PATH
    cfg = C.load_config(cfg_path)
    logger = C.setup_logging(cfg, name="calibrate_taught_flow")

    flow_path = Path(args.flow)

    if args.check:
        return cmd_check(cfg, logger, flow_path)

    flow = load_flow(flow_path)

    # 基础信息必须齐全：夹爪开合值。已有步骤的问题允许通过重录修复。
    for arm in C.ARMS:
        try:
            gripper_values(cfg, arm)
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return 2
    problems, warnings = validate_flow(flow, cfg, require_complete=False)
    for line in warnings:
        print(line)
    if problems:
        print("已有动作流存在问题（下面这些步骤需要重录）：")
        for line in problems:
            print(line)

    indices = target_indices(args)
    stop_event = threading.Event()
    dual: DualArm | None = None
    ok = False
    try:
        dual = DualArm(cfg, logger, stop_event)
        firmware = read_firmware(dual.ports, C.ARMS, logger)
        missing = [a for a in C.ARMS if a not in firmware]
        if missing:
            print(f"无法读取固件标定: {missing}；检查两条臂的串口/电源。",
                  file=sys.stderr)
            return 2

        old_problems, _ = flow_fingerprint_report(flow, firmware)
        if old_problems:
            print("\n已有步骤中有旧电机标定下的点，建议重录；")
            for line in old_problems:
                print("  " + line)

        dual.connect()
        park_arms(dual, cfg, firmware, logger)

        torque_on = {a: True for a in C.ARMS}   # connect 后全部上力
        current_arm = None
        results = {}

        for index in indices:
            d = STEP_DEFS[index - 1]
            arm = d["arm"]
            if current_arm is not None and arm != current_arm:
                if not torque_on[current_arm]:
                    set_joint_torque(dual, current_arm, True)
                    torque_on[current_arm] = True
                    print(f"\n[{current_arm}] 已加力保持在当前位。")
            current_arm = arm

            print_header(index)
            print(f"  夹爪动作: {step_gripper_text(d)}")
            print(f"  当前关节力矩: {'加力' if torque_on[arm] else '已卸力'}")

            # 按预设 pre 状态设置夹爪（只写夹爪，不动关节）。
            try:
                set_gripper_only(dual, cfg, arm, d["pre"])
            except Exception as exc:
                logger.warning("[%s] 设置夹爪 pre=%s 失败: %s",
                               arm, d["pre"], exc)
            try:
                load = dual.arms[arm].read_load()
                pos = dual.arms[arm].read_gripper()
                print(f"  夹爪已设为 {d['pre']}：pos={pos:.1f} load={load:.1f}")
                if d["pre"] == "closed" and load < float(cfg["grasp"]["load_min"]):
                    print("  · 提醒：夹爪负载偏低，请确认方块确实夹在指间。")
            except Exception:
                pass

            if torque_on[arm]:
                print("  即将卸掉 5 个关节的力矩（夹爪继续夹住），请托住该臂。")
                set_joint_torque(dual, arm, False)
                torque_on[arm] = False

            print("  回车记录 / t 加力保持或卸力 / o 张开夹爪 / c 合爪 / s 跳过 / q 退出")
            while True:
                ans = ask("  > ").lower()
                if ans == "q":
                    raise SystemExit(0)
                if ans == "s":
                    print("  跳过。")
                    results[index] = False
                    break
                if ans == "t":
                    try:
                        set_joint_torque(dual, arm, not torque_on[arm])
                        torque_on[arm] = not torque_on[arm]
                        print(f"  关节力矩: {'加力' if torque_on[arm] else '已卸力'}")
                    except Exception as exc:
                        print(f"  ! 切换力矩失败: {exc}")
                    continue
                if ans == "o":
                    try:
                        set_gripper_only(dual, cfg, arm, "open")
                        print(f"  夹爪已张开: load={dual.arms[arm].read_load():.1f}")
                    except Exception as exc:
                        print(f"  ! 张开失败: {exc}")
                    continue
                if ans == "c":
                    try:
                        set_gripper_only(dual, cfg, arm, "closed")
                        print(f"  夹爪已合上: load={dual.arms[arm].read_load():.1f}")
                    except Exception as exc:
                        print(f"  ! 合爪失败: {exc}")
                    continue
                if ans:
                    print("  只支持 回车 / t / o / c / s / q。")
                    continue

                if not record_step(cfg, dual, flow, flow_path, index,
                                   firmware, logger):
                    continue
                apply_post(cfg, dual, index)
                results[index] = True
                if index < len(STEP_DEFS):
                    print("  记录完成。按 t 卸力后，用手把该臂摆到下一步；"
                          "夹爪状态会自动按预设保持。")
                break

        print("\n" + "=" * 72)
        missing_steps = [i for i, d in enumerate(STEP_DEFS, 1)
                         if normalize_flow(flow)["steps"][i - 1].get("q") is None]
        recorded = [i for i in indices if results.get(i)]
        print(f"本次记录: {', '.join(map(str, recorded)) or '无'}")
        print(f"文件: {flow_path}")
        if missing_steps:
            print(f"仍缺步骤: {', '.join(map(str, missing_steps))}")
            print("运行时会被拒绝；请补录后再运行。")
        else:
            print("14 步齐全。下一步:")
            print("  ./run_taught_flow.sh --check")
            print("  ./run_taught_flow.sh --dry-run")
            print("  ./run_taught_flow.sh --minutes 5")
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
        print("\n退出清理会开爪并卸力；如需支持机械臂请现在托住。")
        shutdown(dual, logger)
        logger.info("退出%s", "正常" if ok else "（未完成）")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)
