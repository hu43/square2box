#!/usr/bin/env python
"""calibrate_gripper.py — 单独重新标定夹爪开合值。

只改 config.json 里的两个键：
  - gripper_open   : 夹爪完全张开时的 Present_Position
  - gripper_closed : 夹爪合拢到能夹住方块时的 Present_Position

用法（conda env: hupy3.12）:
  python calibrate_gripper.py --read    # 只读当前配置和夹爪读数，不动
  python calibrate_gripper.py           # 卸力，手摆开/合，回车记录并保存
  python calibrate_gripper.py --test    # 记录后再通电验证开/合

注意：
  - 标定过程会卸力，机械臂会失重，请先托住手臂。
  - “合拢”是合到能夹住方块的程度，不是压到最死；循环运行时会自动测空合值。
  - 只合并写入这两个键，不会改动 hover/grasp/box 等其他标定。
"""

import argparse
import threading
import time

import square2box_loop as S


def main():
    ap = argparse.ArgumentParser(description="重新标定夹爪开合值")
    ap.add_argument("--read", action="store_true", help="只读当前配置和夹爪读数，不卸力")
    ap.add_argument("--test", action="store_true", help="记录后通电验证开/合")
    ap.add_argument("--no-save", action="store_true", help="只打印，不写 config.json")
    args = ap.parse_args()

    cfg = S.load_config()
    logger = S.setup_logging(cfg)
    robot = S.Robot(cfg, threading.Event(), logger)

    try:
        robot.connect()
        old_open = float(cfg["gripper_open"])
        old_closed = float(cfg["gripper_closed"])
        cur = robot.read_gripper()
        load = robot.read_load()
        print(f"配置: gripper_open={old_open:.2f}  gripper_closed={old_closed:.2f}")
        print(f"当前: 夹爪读数={cur:.2f}  负载={load:.1f}")

        if args.read:
            return

        print("\n警告：接下来会卸力，机械臂会失去保持力，请先托住手臂。")
        input("确认手已托住机械臂后按回车卸力... ")
        robot.disable_torque()
        time.sleep(0.3)

        input("1/2 用手把夹爪完全张开，按回车记录 gripper_open... ")
        open_val = robot.read_gripper()
        print(f"  gripper_open = {open_val:.2f}")

        input("2/2 用手把夹爪合拢到能夹住方块的程度，按回车记录 gripper_closed... ")
        closed_val = robot.read_gripper()
        print(f"  gripper_closed = {closed_val:.2f}")

        if open_val <= closed_val:
            print("警告：open <= closed，数值可能反了；仍会保存，请检查夹爪方向。")

        if not args.no_save:
            S.save_config({
                "gripper_open": float(open_val),
                "gripper_closed": float(closed_val),
            })
            print("已写入 config.json（只改这两个键）")
        else:
            print("--no-save：未写入 config.json")

        if args.test:
            input("\n确认手已离开机械臂，按回车通电验证... ")
            robot.enable_torque()
            print("开爪命令...")
            p_open = robot.set_gripper(float(open_val), settle=0.8)
            print(f"  读数={p_open:.2f}")
            print("合爪命令（自由空间，读数可能比标定值更小）...")
            p_closed = robot.set_gripper(float(closed_val), settle=0.8)
            print(f"  读数={p_closed:.2f}  负载={robot.read_load():.1f}")
            robot.set_gripper(float(open_val), settle=0.5)

        print("\n完成。循环脚本下次启动会使用新的开合值。")
    finally:
        robot.disconnect()
        print("已断开并下电力矩；机械臂会失力，请托住。")


if __name__ == "__main__":
    main()
