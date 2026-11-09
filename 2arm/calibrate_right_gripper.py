#!/usr/bin/env python
"""calibrate_right_gripper.py — 修改夹爪开合标定（gripper_open / gripper_closed）。

config.json 里 arms.<arm>.gripper_open / gripper_closed 是夹爪电机的
归一化目标位置（0~100，数值越大越张开）。本脚本只改这两个值，不动
任何位姿，也不需要重标整条臂。

用法（conda activate hupy3.12；先停掉主循环）：
  python calibrate_right_gripper.py --show                    # 看当前值 + 实测开合
  python calibrate_right_gripper.py --show --no-move          # 只读 config，不动电机
  python calibrate_right_gripper.py --open 62 --closed 8      # 直接改成新值
  python calibrate_right_gripper.py --delta-open +3 --delta-closed -1
  python calibrate_right_gripper.py --test                    # 按当前值实际开合一次
  python calibrate_right_gripper.py --measure                 # 手动摆位记录（会卸力）
  python calibrate_right_gripper.py --measure --apply         # 记录并写入

约定：
  * 张开值必须大于闭合值；
  * 两者差太小（<5）夹持行程不够，容易夹不住；
  * --measure 会卸力，需要你用手把夹爪摆到张开/夹住方块的位置再回车。
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import armcore as C
from armdriver import DualArm

DEFAULT_ARM = "right"


def _arm_cfg(cfg: dict, arm: str) -> dict:
    return cfg["arms"][arm]


def show_config(cfg: dict, arm: str) -> tuple[float, float]:
    ac = _arm_cfg(cfg, arm)
    go = ac.get("gripper_open")
    gc = ac.get("gripper_closed")
    print(f"[{arm}] 当前标定: gripper_open={go}  gripper_closed={gc}")
    if go is None or gc is None:
        raise SystemExit(f"[{arm}] 夹爪开合还没标定，先运行 calibrate2arm.py --arm {arm}")
    return float(go), float(gc)


def validate(go: float, gc: float) -> list[str]:
    problems = []
    if not (0.0 <= gc <= 100.0 and 0.0 <= go <= 100.0):
        problems.append("开合值必须在 0~100 之间")
    if gc >= go:
        problems.append(f"闭合值 {gc:.2f} 不小于张开值 {go:.2f}（方向可能反了）")
    if go - gc < 5.0:
        problems.append(f"开合行程只有 {go - gc:.2f}，太小，夹持可能不稳")
    if go > 95.0:
        problems.append(f"张开值 {go:.2f} 接近 100，可能顶到机械限位")
    return problems


def connect(cfg: dict, logger, arm: str) -> tuple[DualArm, threading.Event]:
    stop = threading.Event()
    dual = DualArm(cfg, logger, stop)
    dual.connect()
    return dual, stop


def test_values(cfg: dict, logger, arm: str, go: float, gc: float) -> None:
    dual, stop = connect(cfg, logger, arm)
    a = dual.arms[arm]
    try:
        print(f"[{arm}] 实际张开到 {go:.2f} ...")
        pos_open = a.set_gripper(go, settle=0.45)
        print(f"[{arm}]   实测 pos={pos_open:.2f}")
        print(f"[{arm}] 实际闭合到 {gc:.2f}（夹爪之间请放好方块）...")
        pos_closed = a.set_gripper(gc, settle=0.55)
        load = a.read_load()
        print(f"[{arm}]   实测 pos={pos_closed:.2f}  load={load:.0f}")
        if abs(pos_open - pos_closed) < 5.0:
            print(f"[{arm}] ! 张开/闭合读数几乎一样，夹爪可能没动或被卡住")
    finally:
        try:
            a.set_gripper(go, settle=0.15)
        except Exception:
            pass
        dual.stop_workers()
        try:
            dual.disable_torque()
        except Exception:
            pass
        dual.disconnect()


def measure(cfg: dict, logger, arm: str, apply: bool) -> tuple[float, float] | None:
    """卸力后由人手摆位，回车记录张开/夹住两个读数。"""
    print("=" * 60)
    print("手动测量夹爪开合（全程卸力）")
    print("请确认手臂已放稳/托住，避免卸力后塌下来。")
    print("=" * 60)
    stop = threading.Event()
    dual = DualArm(cfg, logger, stop)
    dual.connect()
    a = dual.arms[arm]
    try:
        a.bus.disable_torque()
        try:
            input("\n把夹爪完全张开，按回车记录 gripper_open... ")
            go = a.read_gripper()
            print(f"  gripper_open = {go:.2f}")
            input("把夹爪合到刚好夹住方块，按回车记录 gripper_closed... ")
            gc = a.read_gripper()
            print(f"  gripper_closed = {gc:.2f}")
        finally:
            # 上力前先把目标写成当前读数，避免夹爪跳回旧目标
            try:
                bus = a.bus.bus
                names = a.bus.joint_names + [a.bus.gripper_name]
                pos = bus.sync_read("Present_Position", names)
                bus.sync_write("Goal_Position",
                               {n: float(pos[n]) for n in names})
                a.bus.enable_torque()
            except Exception:
                pass
        problems = validate(go, gc)
        for p in problems:
            print(f"  ! {p}")
        if apply and not problems:
            C.save_config({"arms": {arm: {"gripper_open": round(float(go), 3),
                                          "gripper_closed": round(float(gc), 3)}}})
            print(f"  已写入 {C.CONFIG_PATH}")
        elif apply:
            print("  有问题，未写入。")
        else:
            print("  未写入（加 --apply 才保存）。")
        return go, gc
    finally:
        dual.stop_workers()
        try:
            dual.disable_torque()
        except Exception:
            pass
        dual.disconnect()


def main() -> int:
    ap = argparse.ArgumentParser(description="修改夹爪开合标定（默认右臂）")
    ap.add_argument("--arm", choices=["left", "right"], default=DEFAULT_ARM)
    ap.add_argument("--open", type=float, default=None, help="新的 gripper_open")
    ap.add_argument("--closed", type=float, default=None, help="新的 gripper_closed")
    ap.add_argument("--delta-open", type=float, default=None, help="在现值上加/减")
    ap.add_argument("--delta-closed", type=float, default=None, help="在现值上加/减")
    ap.add_argument("--show", action="store_true", help="显示当前值（默认行为）")
    ap.add_argument("--test", action="store_true", help="按当前值实际开合一次")
    ap.add_argument("--no-move", action="store_true", help="--show 时不动电机")
    ap.add_argument("--measure", action="store_true", help="卸力后手动测量并记录")
    ap.add_argument("--apply", action="store_true", help="--measure 时写入 config")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = C.load_config(Path(args.config) if args.config else C.CONFIG_PATH)
    logger = C.setup_logging(cfg, name="calib_gripper")
    arm = args.arm
    go, gc = show_config(cfg, arm)

    if args.measure:
        measure(cfg, logger, arm, args.apply)
        return 0

    changed = False
    if args.delta_open is not None:
        go += float(args.delta_open)
        changed = True
    if args.delta_closed is not None:
        gc += float(args.delta_closed)
        changed = True
    if args.open is not None:
        go = float(args.open)
        changed = True
    if args.closed is not None:
        gc = float(args.closed)
        changed = True

    if changed:
        problems = validate(go, gc)
        if problems:
            print("校验失败：")
            for p in problems:
                print(f"  ! {p}")
            return 2
        print(f"[{arm}] 新值: gripper_open={go:.3f}  gripper_closed={gc:.3f}")
        C.save_config({"arms": {arm: {"gripper_open": round(float(go), 3),
                                      "gripper_closed": round(float(gc), 3)}}})
        print(f"已写入 {C.CONFIG_PATH}")

    if args.test:
        test_values(cfg, logger, arm, go, gc)
    elif not changed and not args.show and not args.no_move:
        # 默认 --show 也做一次实测，方便确认值是否还成立
        test_values(cfg, logger, arm, go, gc)
    elif args.show and not args.no_move:
        test_values(cfg, logger, arm, go, gc)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)
