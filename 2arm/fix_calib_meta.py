#!/usr/bin/env python
"""修正 2arm 配置元数据：左臂 home 的 elbow 截断 + 右臂点位补标定指纹。

只改 config.json 的元数据/一个越界 home 值，不动机器人点位本身。
"""
import threading
import logging
import numpy as np
import armcore as C
from armdriver import DualArm

cfg = C.load_config()
log = logging.getLogger("fix")

# 1) 左臂 home elbow 从 +100（截断）挪到 95，仍在标定范围内
home = C.get_pose(cfg, "left", "home")
home = np.asarray(home, float).copy()
home[2] = 95.0
C.set_pose(cfg, "left", "home", home)
print("left home ->", np.round(home, 2).tolist())

# 2) 右臂点位补盖当前电机标定指纹
stop = threading.Event()
dual = DualArm(cfg, log, stop)
try:
    dual.connect(only=["right"])
    bus = dual.arms["right"].bus.bus
    cal = getattr(bus, "calibration", None)
    if not cal:
        raise SystemExit("读不到右臂电机标定")
    short = C.calib_fingerprint_short(cal)
    print("right calib fingerprint =", short)
    for k in C.POSE_KEYS:
        if C.get_pose(cfg, "right", k, required=False) is not None:
            C.set_pose_calib(cfg, "right", k, cal)
finally:
    try:
        dual.disconnect()
    except Exception:
        pass

C.save_config({
    "arms": {
        "left": {"poses": {"home": home.tolist()}},
        "right": {"pose_calib": C.get_pose_calib(cfg, "right")},
    }
})
print("saved config.json")
