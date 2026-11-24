#!/usr/bin/env python
"""calibrate_handoff.py — 一次性示教两个交叉交接位（无摄像头）。

记录两个位姿：
  left.place_right  左臂把夹爪摆到“右臂 table_pick”的位置和抓取高度
  right.place_left  右臂把夹爪摆到“左臂 table_pick”的位置和抓取高度

这两个位姿既是放置位，也能被同一条臂用来在对方交接点抓取。
运行主程序时会**原封不动**地发送这两个位姿（以及已有 10 个标定点），
不会叠加释放抬高、pan/深度偏移或反推 hover。
不同臂的关节值不能互换，所以必须实测，不能用对方的 table_pick。

用法（conda activate hupy3.12）：
    cd /home/zhoulin/hu/square2box/1
    python calibrate_handoff.py                 # 依次示教两个点
    python calibrate_handoff.py --arm left      # 只示教 left.place_right
    python calibrate_handoff.py --check         # 只读校验

操作：
    1. 程序先把参考臂移到它的 table_hover，让你看清“它的 table_pick 正下方”
       就是交接点，然后参考臂回 home 让开。
    2. 用要示教的那条臂：卸力后用手把夹爪摆到方块上，两指包住方块，
       高度与参考臂抓取时一致（不要压桌面）。
    3. 回车记录；t 加力保持/卸力；s 跳过；q 退出。

安全：示教臂卸力时请托住；退出会开爪并卸力。
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np

import armcore as C
from armdriver import DualArm
from handoff_core import (
    CROSS_KEY,
    CROSS_FRONT,
    OWN_KEY,
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


def show_reference(dual: DualArm, arm: str, cfg: dict, logger) -> None:
    """把参考臂移到 table_hover 再回 home，给操作者看清交接点。"""
    a = dual.arms[arm]
    open_g = float(cfg["arms"][arm]["gripper_open"])
    try:
        logger.info("[%s] 参考点演示：去 table_hover", arm)
        a.move(["table_approach", "table_hover"], gripper=open_g,
               label=f"{arm} 参考点", speed_scale=0.7, timeout=10.0)
        print(f"\n  参考点：{arm} 臂当前 table_hover 的【正下方】就是它的 "
              f"table_pick 交接点。")
        print("  请把方块放在那个位置（桌面上、该臂正前方）。")
        a.move(["table_approach", "carry", "home"], gripper=open_g,
               label=f"{arm} 参考点让开", speed_scale=0.7, timeout=10.0)
    except Exception as exc:
        logger.warning("[%s] 参考点演示未完成: %s", arm, exc)
        print(f"  （{arm} 参考点演示未完成，请按它已标定的 table_pick 位置放方块。）")


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


def record_pose(cfg: dict, dual: DualArm, arm: str, key: str,
                cfg_path: Path, logger) -> bool:
    a = dual.arms[arm]
    q = np.asarray(a.read_joints(), dtype=float)

    saturation = C.saturated_joints(q)
    margins = pose_margins(q)
    tight = [f"{n}({m:.1f})" for n, m in margins if m < 1.0]
    near = [f"{n}({m:.1f})" for n, m in margins if 1.0 <= m < 2.0]
    if saturation or tight:
        print(f"  ! 当前位姿不可用：截断={saturation or '无'}，贴边={tight or '无'}")
        print("  请把交接点往该臂方向挪一点、或调整腕部姿态，再按回车重试。")
        return False
    if near:
        print(f"  · 提醒：这些关节离边界很近: {', '.join(near)}（仍可记录）")

    own = C.get_pose(cfg, arm, OWN_KEY, required=False)
    if own is not None and float(np.abs(q - own).max()) < 2.0:
        print("  · 提醒：这个交叉位与自身 table_pick 几乎重合；"
              "请确认两个前方交接点确实是分开的。")

    bus = getattr(a.bus, "bus", None)
    cal = getattr(bus, "calibration", None) if bus is not None else None
    if not cal:
        print("  ! 读不到电机标定文件，无法记录指纹。")
        return False

    C.set_pose(cfg, arm, key, q)
    C.set_pose_calib(cfg, arm, key, cal)
    saved_q = cfg["arms"][arm]["poses"][key]
    saved_fp = cfg["arms"][arm]["pose_calib"][key]
    C.save_config({"arms": {arm: {
        "poses": {key: saved_q},
        "pose_calib": {key: saved_fp},
    }}}, cfg_path)

    print(f"\n  已保存 [{arm}].{key} = {C.fmt_q(q)}")
    print(f"  指纹 {saved_fp}  ->  {cfg_path}")
    return True


def calibrate_one(cfg: dict, dual: DualArm, active: str, key: str,
                  cfg_path: Path, logger) -> bool:
    ref = CROSS_FRONT[active]
    print("\n" + "=" * 70)
    print(f"示教 [{active}].{key}：{active} 臂放到 {ref} 臂前方")
    print("=" * 70)

    show_reference(dual, ref, cfg, logger)

    a = dual.arms[active]
    open_g = float(cfg["arms"][active]["gripper_open"])
    print(f"\n  接下来会卸力 {active} 臂，请先托住它。")
    ask("  准备好后按回车继续（q 退出）")
    try:
        a.move(["carry"], gripper=open_g, label=f"{active} 准备示教",
               speed_scale=0.6, timeout=10.0)
    except Exception as exc:
        logger.warning("[%s] 去 carry 未完成: %s", active, exc)
    try:
        a.open_gripper()
    except Exception:
        pass
    a.bus.disable_torque()
    torque_on = False
    print(f"  {active} 已卸力。用手把夹爪摆到方块上：两指包住方块，"
          f"高度与 {ref} 臂抓取时一致，不要压桌面。")
    print("  回车记录 / t 加力保持或卸力 / s 跳过 / q 退出")

    while True:
        ans = ask("  > ")
        if ans.lower() == "q":
            raise SystemExit(0)
        if ans.lower() == "s":
            print("  跳过。")
            return False
        if ans.lower() == "t":
            if torque_on:
                a.bus.disable_torque()
                torque_on = False
                print("  已卸力，请托住。")
            else:
                if set_torque_on(a, logger):
                    torque_on = True
                    print("  已加力，可以松手；按 t 再卸力。")
            continue
        if ans:
            print("  只支持 回车 / t / s / q。")
            continue
        if record_pose(cfg, dual, active, key, cfg_path, logger):
            # 记录后先加力保持，再回 home，避免松手时塌下去。
            if not torque_on:
                if set_torque_on(a, logger):
                    torque_on = True
                    print("  已加力保持。")
            try:
                a.move(["carry", "home"], gripper=open_g,
                       label=f"{active} 示教后回 home",
                       speed_scale=0.6, timeout=10.0)
            except Exception as exc:
                logger.warning("[%s] 示教后回 home 未完成: %s", active, exc)
            return True


def cmd_check(cfg: dict, logger) -> int:
    print("=" * 70)
    print("只读校验（--check：不使能力矩，不发送运动指令）")
    print("=" * 70)
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

    for line in warnings:
        print(line)
    if problems:
        print("\n阻断问题：")
        for line in problems:
            print(line)
        print(f"\n校验失败：{len(problems)} 个阻断问题。")
        return 1
    print("\n校验通过：基础点位和两个交叉位都可用。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="一次性示教双臂交叉交接位")
    ap.add_argument("--arm", choices=("left", "right", "both"), default="both",
                    help="示教哪条臂的交叉位（默认两个都示教）")
    ap.add_argument("--config", default=None, help="config.json 路径")
    ap.add_argument("--check", action="store_true",
                    help="只读校验，不动电机")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else C.CONFIG_PATH
    cfg = C.load_config(cfg_path)
    logger = C.setup_logging(cfg, name="calibrate_handoff")

    if args.check:
        return cmd_check(cfg, logger)

    problems, warnings = validate_config(cfg, require_cross=False)
    for line in warnings:
        logger.warning("%s", line)
    if problems:
        for line in problems:
            logger.error("%s", line)
        print("基础点位不完整或不可用，先重跑 calibrate2arm.py。", file=sys.stderr)
        return 2

    arms = ["left", "right"] if args.arm == "both" else [args.arm]
    dual: DualArm | None = None
    ok = False
    try:
        dual = DualArm(cfg, logger, threading.Event())
        firmware = read_firmware(dual.ports, C.ARMS, logger)
        missing = [a for a in C.ARMS if a not in firmware]
        if missing:
            print(f"无法读取固件标定: {missing}；检查两条臂的串口/电源。",
                  file=sys.stderr)
            return 2
        fp_problems, fp_warnings = fingerprint_report(
            cfg, firmware, include_cross=False)
        for line in fp_warnings:
            logger.warning("%s", line)
        if fp_problems:
            for line in fp_problems:
                logger.error("%s", line)
            print("当前 10 个点位不是在当前电机标定下录的；请先完整重跑 "
                  "calibrate2arm.py（两条臂），再回来示教交叉位。",
                  file=sys.stderr)
            return 2

        dual.connect()
        results = {}
        for active in arms:
            key = CROSS_KEY[active]
            existing = (cfg["arms"][active].get("poses") or {}).get(key)
            if existing is not None:
                print(f"\n[{active}].{key} 已存在: {C.fmt_q(existing)}")
                if ask("  回车重新示教 / s 保留旧值: ").lower() == "s":
                    results[active] = False
                    continue
            results[active] = calibrate_one(
                cfg, dual, active, key, cfg_path, logger)

        print("\n" + "=" * 70)
        for active, saved in results.items():
            print(f"  [{active}].{CROSS_KEY[active]}: "
                  f"{'已记录' if saved else '未记录/保留旧值'}")
        print("\n下一步：")
        print("  ./run_handoff.sh --check")
        print("  ./run_handoff.sh --dry-run")
        print("  ./run_handoff.sh --minutes 5")
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
