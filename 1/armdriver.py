#!/usr/bin/env python
"""armdriver — 双臂（Feetech STS3215）驱动层。

设计要点（针对“流畅、少停顿”）：
  * 每条臂一个 worker 线程，轨迹在后台按 100Hz 绝对时基发送；
    两条臂的轨迹可以完全并行执行，不需要等一条走完再动另一条。
  * 轨迹一次算好（限速时间参数化 + 两端缓起缓停），循环里只做插值，无额外计算。
  * 只在目标值变化超过阈值时才下发，减少总线占用。
  * 到位判定只读一次（带重试），不再每步都同步读总线。
  * 串口掉线 → 计数 → 自动重连，重连后重新使能力矩。

单位：关节 [-100, 100]（归一化角度），夹爪 [0, 100]，0=闭合。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from armcore import (
    ARM_PREFIX,
    BusError,
    JOINTS,
    NJ,
    StopRequested,
    build_path,
    calibration_paths,
    clamp,
    expand_path,
    official_calibrate_cmd,
    smootherstep,
    time_parameterize,
    vmax_vector,
)

try:
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, "/home/zhoulin/hu/lerobot/src")
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus


PROBE_IDS = list(range(1, 9))


# ---------------------------------------------------------------------------
# 串口发现
# ---------------------------------------------------------------------------

def list_acm_ports() -> list[str]:
    ports = sorted(Path("/dev").glob("ttyACM*"))
    return [str(p) for p in ports]


def probe_port_counts(ports: Optional[Sequence[str]] = None) -> dict[str, int]:
    """对每个串口 Ping id 1..8，返回每个口上应答的电机数量。"""
    ports = list(ports or list_acm_ports())
    result: dict[str, int] = {}
    for port in ports:
        motors = {f"probe_{i}": Motor(i, "sts3215", MotorNormMode.RANGE_M100_100)
                  for i in PROBE_IDS}
        bus = None
        try:
            bus = FeetechMotorsBus(port=port, motors=motors)
            bus.connect(handshake=False)
            found = 0
            for i in PROBE_IDS:
                try:
                    if bus.ping(i) is not None:
                        found += 1
                except Exception:
                    pass
            result[port] = found
        except Exception:
            result[port] = -1
        finally:
            if bus is not None:
                try:
                    ser = getattr(bus.port_handler, "ser", None)
                    if ser is not None:
                        ser.reset_input_buffer()
                except Exception:
                    pass
                try:
                    bus.disconnect(disable_torque=False)
                except Exception:
                    pass
    return result


def resolve_arm_ports(cfg: dict, logger=None) -> dict[str, str]:
    """解析 left/right 串口。

    优先使用 config 里显式写的端口；写 "auto" 或端口不存在时，
    按“电机数量多的是左臂（含头，8 个）”自动识别。
    """
    dev = cfg["devices"]
    wanted = {"left": str(dev.get("left_port") or "auto"),
              "right": str(dev.get("right_port") or "auto")}

    def ok(p: str) -> bool:
        return p and p != "auto" and Path(p).exists()

    if ok(wanted["left"]) and ok(wanted["right"]) and wanted["left"] != wanted["right"]:
        return wanted

    counts = probe_port_counts()
    usable = {p: n for p, n in counts.items() if n >= 6}
    if logger:
        logger.info("串口探测结果: %s", counts)
    if len(usable) < 2:
        raise BusError(
            f"只找到 {len(usable)} 条可用机械臂串口（{counts}）。"
            "检查 USB 线和电源，或修好 /etc/udev/rules.d/99-xlerobot.rules 后重插。")
    left = max(usable, key=lambda p: usable[p])
    right = next(p for p in usable if p != left)
    return {"left": left, "right": right}


# ---------------------------------------------------------------------------
# 标定
# ---------------------------------------------------------------------------

def load_arm_calibration(path, arm: str | None = None) -> dict[str, MotorCalibration]:
    """读官方 lerobot-calibrate 写出的标定文件。

    官方文件里的键是裸名（shoulder_pan / gripper ...），本程序的电机表
    用的是带前缀的名字（left_arm_shoulder_pan ...）。两种都认：先按带前缀
    的找，找不到再按裸名找，然后统一转成带前缀的名字。

    只保留该臂自己的 6 个电机条目；文件里如果同时装着其它臂/头部的条目
    （老的单文件格式）就忽略掉。arm 为 None 时返回文件里的全部条目。
    """
    p = expand_path(path)
    if not p.is_file():
        raise SystemExit(
            f"找不到电机标定文件: {p}\n"
            f"请先用官方工具标定这条臂（电机行程一律用官方标定）:\n"
            f"  {official_calibrate_cmd(arm) if arm else 'lerobot-calibrate ...'}")
    raw = json.loads(p.read_text())

    def to_cal(d) -> MotorCalibration:
        return MotorCalibration(
            id=int(d["id"]),
            drive_mode=int(d.get("drive_mode", 0)),
            homing_offset=int(d.get("homing_offset", 0)),
            range_min=int(d["range_min"]),
            range_max=int(d["range_max"]),
        )

    if arm is None:
        out: dict[str, MotorCalibration] = {}
        for name, d in raw.items():
            try:
                out[name] = to_cal(d)
            except Exception:
                continue
        return out

    prefix = ARM_PREFIX[arm]
    out = {}
    missing = []
    for motor_key, short in ([(j, j) for j in JOINTS] + [("gripper", "gripper")]):
        d = raw.get(f"{prefix}_{short}", raw.get(short))
        if d is None:
            missing.append(f"{prefix}_{short}")
            continue
        try:
            out[f"{prefix}_{short}"] = to_cal(d)
        except Exception:
            missing.append(f"{prefix}_{short}")
    if missing:
        raise SystemExit(
            f"{p} 缺少 {arm} 的条目: {missing}\n"
            f"请重跑官方标定: {official_calibrate_cmd(arm)}")
    return out


def calibration_from_bus(bus: FeetechMotorsBus) -> dict[str, MotorCalibration]:
    """读电机固件里的标定（Homing_Offset + 限位），用于和官方文件做核对。

    注意：**电机行程的唯一来源是官方 lerobot-calibrate 写出的文件**，
    这个函数只用来检查固件是否被写坏了（比如标定中途 Ctrl+C，文件写好了
    但固件没写完，限位会退化成 min == max，关节直接锁死）。
    """
    out: dict[str, MotorCalibration] = {}
    for name, m in bus.motors.items():
        out[name] = MotorCalibration(
            id=m.id,
            drive_mode=0,
            homing_offset=int(bus.read("Homing_Offset", name, normalize=False)),
            range_min=int(bus.read("Min_Position_Limit", name, normalize=False)),
            range_max=int(bus.read("Max_Position_Limit", name, normalize=False)),
        )
    return out


def make_motors(prefix: str) -> dict[str, Motor]:
    """一条臂的 6 个电机（5 关节 + 夹爪）。

    左臂串口上通常还挂着 id 7/8 的头部电机，但本程序从不读写它们，
    所以不放进电机表：放进来会要求标定文件里也有 head_motor_1/2 的
    条目，而 lerobot 的 xlerobot.json 只存 12 条手臂标定，导致整条
    左臂连接失败。总线只访问表里的 id，忽略 7/8 不影响通信。
    """
    motors = {}
    for i, j in enumerate(JOINTS, start=1):
        motors[f"{prefix}_{j}"] = Motor(i, "sts3215", MotorNormMode.RANGE_M100_100)
    motors[f"{prefix}_gripper"] = Motor(6, "sts3215", MotorNormMode.RANGE_0_100)
    return motors


# ---------------------------------------------------------------------------
# 单臂
# ---------------------------------------------------------------------------

@dataclass
class MotionCommand:
    path: np.ndarray
    times: np.ndarray
    gripper: Optional[float]
    label: str
    gripper_from: Optional[float] = None
    gripper_to: Optional[float] = None
    switch_frac: float = 0.3
    gripper_profile: Optional[np.ndarray] = None
    id: int = 0


def _gripper_ramp(n: int, times: np.ndarray, g0: float, g1: float,
                  switch_frac: float) -> np.ndarray:
    """在轨迹的前 switch_frac 时间内把夹爪从 g0 过渡到 g1。

    这样“松开”可以和“抬起”同时发生，省掉一段原地等待。
    """
    total = float(times[-1]) if len(times) else 0.0
    frac = np.clip(switch_frac, 0.02, 1.0)
    span = max(total * frac, 0.05)
    out = np.empty(n, dtype=float)
    for i in range(n):
        s = smootherstep(min(1.0, float(times[i]) / span))
        out[i] = g0 + (g1 - g0) * s
    return out


class ArmBus:
    """一条臂的串口封装（线程安全）。"""

    def __init__(self, arm: str, cfg: dict, port: str, calibration: dict,
                 logger, stop_event: threading.Event,
                 calibration_path: str | None = None):
        self.arm = arm
        self.cfg = cfg
        self.port = port
        self.prefix = ARM_PREFIX[arm]
        self.log = logger
        self.stop_event = stop_event
        self.calibration = calibration
        self.calibration_path = calibration_path
        self.bus: Optional[FeetechMotorsBus] = None
        self.lock = threading.RLock()
        self.joint_names = [f"{self.prefix}_{j}" for j in JOINTS]
        self.gripper_name = f"{self.prefix}_gripper"
        self._errors = 0
        self.gripper_cmd: float = 100.0

    # ---- 生命周期 ----
    @staticmethod
    def _flush_rx(bus: Optional[FeetechMotorsBus]) -> None:
        """丢掉串口接收缓冲里的残留字节。

        端口探测（ping 8 个 id）和上一个进程都可能留下没读完的应答包。
        重开端口后这些旧字节会被当成新回包，报
        "Incorrect status packet"，所以每次写失败、以及刚连上时都清一次。
        """
        if bus is None:
            return
        try:
            ser = getattr(getattr(bus, "port_handler", None), "ser", None)
            if ser is not None:
                ser.reset_input_buffer()
        except Exception:
            pass

    def _write_retry(self, bus: Optional[FeetechMotorsBus], reg: str, name: str,
                     value: int, tries: Optional[int] = None) -> bool:
        """写单个寄存器，失败就清缓冲重发。返回是否成功。"""
        if bus is None:
            return False
        tries = tries if tries is not None else int(self.cfg["safety"].get("bus_retries", 2)) + 2
        last = None
        for _ in range(max(1, tries)):
            try:
                bus.write(reg, name, value)
                return True
            except Exception as exc:
                last = exc
                self._flush_rx(bus)
                time.sleep(0.03)
        self.log.debug("[%s] 写 %s@%s=%s 失败: %s", self.arm, reg, name, value, last)
        return False

    def connect(self) -> None:
        with self.lock:
            self.disconnect()
            motors = make_motors(self.prefix)
            cal = {n: c for n, c in self.calibration.items() if n in motors}
            missing = [n for n in motors if n not in cal]
            if missing:
                raise BusError(f"[{self.arm}] 标定缺少 {missing}")
            bus = FeetechMotorsBus(port=self.port, motors=motors, calibration=cal)
            bus.connect(handshake=False)
            self.bus = bus
            body = self.joint_names + [self.gripper_name]
            # 刚打开端口时先清掉残留字节，再等总线稳定
            self._flush_rx(bus)
            time.sleep(0.05)
            self._flush_rx(bus)
            # EPROM 参数（P/D/I）必须在卸力时写
            for name in body:
                self._write_retry(bus, "Torque_Enable", name, 0)
            bad = []
            for name in body:
                ok = self._write_retry(bus, "Operating_Mode", name, 0)  # 位置模式
                ok &= self._write_retry(bus, "Return_Delay_Time", name, 0)
                mo = self.cfg["motion"]
                if name in self.joint_names:
                    # 出厂 20 的刚度在带负载时会有 1.5° 左右的静差，同一个
                    # 姿态每次停的位置都不一样，抓取就在成败边缘。提到 40。
                    ok &= self._write_retry(bus, "P_Coefficient", name,
                                            int(mo.get("p_gain", 40)))
                    ok &= self._write_retry(bus, "I_Coefficient", name,
                                            int(mo.get("i_gain", 0)))
                    ok &= self._write_retry(bus, "D_Coefficient", name,
                                            int(mo.get("d_gain", 40)))
                else:
                    ok &= self._write_retry(bus, "P_Coefficient", name,
                                            int(mo.get("gripper_p_gain", 32)))
                    ok &= self._write_retry(bus, "D_Coefficient", name,
                                            int(mo.get("gripper_d_gain", 40)))
                    # 夹爪出厂扭矩上限只有 500（关节是 1000），夹塑料方块
                    # 抬起来就滑。Max_Torque_Limit 是 EEPROM，必须在
                    # enable_torque 写 Lock=1 之前改。
                    ok &= self._write_retry(bus, "Max_Torque_Limit", name,
                                            int(mo.get("gripper_max_torque", 1000)))
                    ok &= self._write_retry(bus, "Torque_Limit", name,
                                            int(mo.get("gripper_torque", 1000)))
                if not ok:
                    bad.append(name)
            if bad:
                self.log.warning("[%s] 这些电机参数写入失败（其余正常）: %s", self.arm, bad)
            # 加速度是 SRAM，可以随时写；不支持的固件忽略
            acc = int(self.cfg["motion"].get("acceleration", 150))
            for name in body:
                self._write_retry(bus, "Maximum_Acceleration", name, 254)
                self._write_retry(bus, "Acceleration", name, acc)
            self._errors = 0
            # 电机行程一律以官方 lerobot-calibrate 写出的文件为准（用户要求）。
            # 固件只做一致性检查：官方工具写固件和写文件是两步，中间被
            # Ctrl+C 打断就会留下“文件是好的、固件是坏的”，而限位退化
            # （min == max）时电机会把关节锁死，表现成“发指令但不动”。
            self._verify_firmware_calibration(bus)
            self.log.info("[%s] 已连接 %s", self.arm, self.port)

    def _verify_firmware_calibration(self, bus: FeetechMotorsBus) -> None:
        """核对电机固件里的限位与官方标定文件是否一致。

        这两种都是确定性的配置问题，抛 SystemExit 让进程退出（而不是当
        成总线抖动一直重试），否则 24/7 常驻服务会卡在一个永远好不了的
        状态里空转：
          - 固件退化成 min == max：关节根本不会动；
          - 固件限位与文件不一致：归一化用文件、执行用固件，臂会在中途
            被截断，抓取永远差一截——无人值守时看不出来。
        """
        try:
            fw = calibration_from_bus(bus)
        except Exception as exc:
            self.log.warning("[%s] 读电机固件标定失败: %s", self.arm, exc)
            return
        degenerated, differs = [], []
        for name, got in fw.items():
            want = self.calibration.get(name)
            if want is None:
                continue
            if got.range_max - got.range_min < 100:
                degenerated.append(name)
            elif (want.range_min != got.range_min
                  or want.range_max != got.range_max
                  or want.homing_offset != got.homing_offset):
                differs.append(name)
        short = lambda n: n.replace(self.prefix + "_", "")
        if degenerated:
            detail = ", ".join(
                f"{short(n)}({fw[n].range_min}..{fw[n].range_max})"
                for n in degenerated)
            raise SystemExit(
                f"[{self.arm}] 电机固件里的行程已退化，关节会被锁死动不了: {detail}\n"
                f"标定文件是好的（{self.calibration_path}），但没写进电机。\n"
                f"重跑官方标定，在提示处直接按【回车】把文件写回电机:\n"
                f"  {official_calibrate_cmd(self.arm)}")
        if differs:
            detail = ", ".join(short(n) for n in differs)
            raise SystemExit(
                f"[{self.arm}] 电机固件限位与标定文件不一致: {detail}\n"
                f"程序按文件归一化、电机按固件限位执行，臂会在中途被截断，\n"
                f"抓取位置会一直差一截。重跑官方标定，在提示处直接按【回车】\n"
                f"把文件写回电机:\n"
                f"  {official_calibrate_cmd(self.arm)}")
        self.log.debug("[%s] 固件限位与标定文件一致", self.arm)


    def disconnect(self) -> None:
        with self.lock:
            bus, self.bus = self.bus, None
            if bus is None:
                return
            try:
                bus.disconnect(disable_torque=False)
            except Exception:
                pass

    def enable_torque(self) -> None:
        """逐电机上力。单个电机丢包不影响其它电机，也不用重启整个程序。

        lerobot 的 enable_torque 内部 num_retry=0，一个包坏掉就抛异常；
        这里自己按电机重试，只有“一个都没上力”才报错。
        """
        with self.lock:
            bus = self.bus
            if bus is None:
                raise BusError(f"[{self.arm}] 未连接")
            body = self.joint_names + [self.gripper_name]
            ok_names = []
            failed = []
            for name in body:
                if self._write_retry(bus, "Torque_Enable", name, 1):
                    ok_names.append(name)
                else:
                    failed.append(name)
            # Lock 只是防误写 EPROM，写不上不影响运动，失败也无所谓
            for name in body:
                self._write_retry(bus, "Lock", name, 1, tries=1)
            if not ok_names:
                raise BusError(f"[{self.arm}] 全部电机上力失败: {failed}")
            if failed:
                self.log.warning("[%s] 这些电机上力失败（下次重连会再试）: %s",
                                 self.arm, failed)
            else:
                self.log.debug("[%s] 6 个电机已上力", self.arm)

    def disable_torque(self) -> None:
        with self.lock:
            bus = self.bus
            if bus is None:
                return
            for name in self.joint_names + [self.gripper_name]:
                # 卸力失败就多试几次，宁可慢一点也不能让臂带着力
                self._write_retry(bus, "Torque_Enable", name, 0, tries=3)

    # ---- 重试/错误计数 ----
    def _call(self, fn, *args, **kwargs):
        last = None
        retries = int(self.cfg["safety"].get("bus_retries", 2))
        for _ in range(retries + 1):
            try:
                out = fn(*args, **kwargs)
                self._errors = 0
                return out
            except Exception as exc:  # 通信类异常
                last = exc
                self._errors += 1
                time.sleep(0.02)
        raise BusError(f"[{self.arm}] 总线错误: {last}")

    @property
    def error_streak(self) -> int:
        return self._errors

    # ---- 读 ----
    def read_state(self) -> tuple[np.ndarray, float]:
        with self.lock:
            if self.bus is None:
                raise BusError(f"[{self.arm}] 未连接")
            obs = self._call(self.bus.sync_read, "Present_Position",
                             self.joint_names + [self.gripper_name])
            q = np.array([float(obs[n]) for n in self.joint_names], dtype=float)
            return q, float(obs[self.gripper_name])

    def read_joints(self) -> np.ndarray:
        return self.read_state()[0]

    def read_gripper(self) -> float:
        return self.read_state()[1]

    def read_load(self) -> float:
        with self.lock:
            if self.bus is None:
                return 0.0
            for reg in ("Present_Load", "Present_Current"):
                try:
                    obs = self._call(self.bus.sync_read, reg, [self.gripper_name])
                    return abs(float(obs[self.gripper_name]))
                except Exception:
                    continue
            return 0.0

    def read_temp(self) -> float:
        with self.lock:
            if self.bus is None:
                return 0.0
            try:
                obs = self._call(self.bus.sync_read, "Present_Temperature", self.joint_names)
            except Exception:
                return 0.0
        vals = []
        for v in obs.values():
            try:
                f = float(v)
            except Exception:
                continue
            if 0.0 <= f <= 110.0:
                vals.append(f)
        return max(vals) if vals else 0.0

    # ---- 写 ----
    def write_goal(self, q: np.ndarray, gripper: Optional[float] = None) -> None:
        with self.lock:
            if self.bus is None:
                raise BusError(f"[{self.arm}] 未连接")
            if gripper is not None:
                self.gripper_cmd = float(clamp(gripper, 0.0, 100.0))
            values = {n: float(clamp(v, -100.0, 100.0))
                      for n, v in zip(self.joint_names, q)}
            values[self.gripper_name] = self.gripper_cmd
            self._call(self.bus.sync_write, "Goal_Position", values)


class Arm:
    """单臂高层接口：后台线程执行轨迹，支持并行与即时停下。"""

    def __init__(self, arm: str, cfg: dict, bus: ArmBus, logger,
                 stop_event: threading.Event):
        self.arm = arm
        self.cfg = cfg
        self.bus = bus
        self.log = logger
        self.stop_event = stop_event

        self._queue: list[MotionCommand] = []
        self._cmd_id = 0
        self._done_id = 0
        self._expect_id = 0
        self._cond = threading.Condition()
        self._last_cmd_q: Optional[np.ndarray] = None
        self._last_cmd_g: Optional[float] = None
        # 最后一条“已提交但可能还没执行”的指令终点。预位是非阻塞的，提交新
        # 指令时臂可能还在飞往预位点；若用实际位置 hold_q 起插值，新轨迹会
        # 在执行时先跳回旧位置。用计划终点续接才连续。
        self._planned_q: Optional[np.ndarray] = None
        self._worker = threading.Thread(target=self._loop, daemon=True,
                                        name=f"arm-{arm}")
        self._worker.start()
        self.hold_q: Optional[np.ndarray] = None
        self.hold_gripper: Optional[float] = None
        self.last_motion_s = 0.0
        # 最近一条指令的执行异常；wait() 据此返回 False，让调用方别把
        # “没走到位”当成“到位了”。
        self.last_error: Optional[BaseException] = None

    # ---- worker ----
    def _loop(self) -> None:
        last_q = None
        last_g = None
        while not self.stop_event.is_set():
            with self._cond:
                self._cond.wait_for(
                    lambda: bool(self._queue) or self.stop_event.is_set(), timeout=0.5)
                if not self._queue:
                    continue
                cmd = self._queue.pop(0)
                cid = cmd.id
            try:
                self._execute(cmd, last_q, last_g)
                with self._cond:
                    last_q = self._last_cmd_q
                    last_g = self._last_cmd_g
            except StopRequested:
                return
            except Exception as exc:
                # 必须留痕：以前这里只记一条日志就把指令标成完成，调用方
                # 的 wait() 照样返回，于是臂没走到位也继续去闭合夹爪。
                self.log.error("[%s] 轨迹执行失败 (%s): %s", self.arm, cmd.label, exc)
                self.last_error = exc
                self._errors_in_worker = getattr(self, "_errors_in_worker", 0) + 1
            finally:
                with self._cond:
                    self._done_id = cid
                    self._cond.notify_all()

    def _execute(self, cmd: MotionCommand, last_q, last_g) -> None:
        times, path = cmd.times, cmd.path
        hz = float(self.cfg["motion"]["control_hz"])
        gripper = cmd.gripper if cmd.gripper is not None else last_g
        t0 = time.perf_counter()
        sent_q = None if last_q is None else np.asarray(last_q, dtype=float)
        sent_g = None if last_g is None else float(last_g)
        n = len(times)
        for i in range(n):
            if self.stop_event.is_set():
                raise StopRequested()
            target_t = t0 + times[i]
            dt = target_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            q = path[i]
            g = gripper
            if cmd.gripper_profile is not None:
                g = float(cmd.gripper_profile[i])
            # 值变化太小就不发，节省总线时间
            if sent_q is not None and float(np.abs(q - sent_q).max()) < 0.02:
                if g is None or (sent_g is not None and abs(g - sent_g) < 0.05):
                    continue
            self.bus.write_goal(q, g)
            sent_q, sent_g = q, (g if g is not None else sent_g)
        if sent_q is None:
            sent_q = path[-1]
        if sent_g is not None:
            self.bus.gripper_cmd = sent_g
        self._last_cmd_q = sent_q.copy()
        self._last_cmd_g = sent_g
        self.hold_q = sent_q.copy()
        if sent_g is not None:
            self.hold_gripper = float(sent_g)
            self.bus.gripper_cmd = float(sent_g)
        self.last_motion_s = time.perf_counter() - t0
        time.sleep(float(self.cfg["motion"].get("settle_s", 0.05)))

    # ---- 对外 ----
    def start(self, waypoints: Sequence[Sequence[float]], *,
              gripper: Optional[float] = None, label: str = "",
              speed_scale: float = 1.0, ease_frac: Optional[float] = None,
              precision: bool = False) -> float:
        """把一条轨迹交给后台线程执行（不阻塞）。返回预估时长（秒）。"""
        return self._submit(waypoints, gripper=gripper, label=label,
                            speed_scale=speed_scale, ease_frac=ease_frac,
                            precision=precision)

    def start_mixed(self, waypoints: Sequence[Sequence[float]], *,
                    gripper_from: Optional[float], gripper_to: float,
                    switch_frac: float = 0.3, label: str = "",
                    speed_scale: float = 1.0, ease_frac: Optional[float] = None,
                    precision: bool = False) -> float:
        """沿轨迹运动的同时把夹爪从 gripper_from 连续变到 gripper_to。"""
        return self._submit(waypoints, gripper=None, label=label,
                            speed_scale=speed_scale, ease_frac=ease_frac,
                            precision=precision, gripper_from=gripper_from,
                            gripper_to=gripper_to, switch_frac=switch_frac)

    def _submit(self, waypoints, *, gripper, label, speed_scale, ease_frac, precision,
                gripper_from=None, gripper_to=None, switch_frac=0.3) -> float:
        pts = np.asarray(waypoints, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != NJ:
            raise ValueError(f"waypoints 形状错误: {pts.shape}")
        pts = np.clip(pts, -100.0, 100.0)
        # 从“上一条指令的终点”起插值：单点位调用也能得到平滑轨迹，
        # 而不是一步跳到目标让电机内部硬插值。用 _planned_q 而不是实际
        # 位置，是因为预位是非阻塞的——提交时臂可能还在路上，用实际位置
        # 会在轨迹执行时先跳回去。
        start = self._planned_q if self._planned_q is not None else self.hold_q
        if start is not None and float(np.abs(pts[0] - start).max()) > 0.05:
            pts = np.vstack([np.asarray(start, float)[None, :], pts])
        mo = self.cfg["motion"]
        scale = float(speed_scale)
        if precision:
            scale *= float(mo["precision_scale"])
        vmax = vmax_vector(self.cfg) * max(0.05, scale) * float(mo["cruise_scale"])
        path = build_path(pts, max_overshoot=float(mo["max_overshoot"]))
        times, path = time_parameterize(
            path, vmax, ease_frac=float(mo["ease_frac"] if ease_frac is None else ease_frac))
        profile = None
        if gripper_to is not None:
            g0 = float(gripper_from) if gripper_from is not None else self.bus.gripper_cmd
            profile = _gripper_ramp(len(times), times, g0, float(gripper_to), switch_frac)
        with self._cond:
            self._cmd_id += 1
            cid = self._cmd_id
            cmd = MotionCommand(
                path=path, times=times,
                gripper=None if gripper is None else float(gripper),
                label=label or "motion",
                gripper_from=gripper_from, gripper_to=gripper_to,
                switch_frac=switch_frac, gripper_profile=profile)
            cmd.id = cid
            # 队列只保留“待执行”的指令；正常情况下调用方都 start+wait，
            # 只有预位时才可能叠加，保持顺序执行是正确行为。
            self._queue.append(cmd)
            self._planned_q = path[-1].copy()
            self._expect_id = cid
            self.last_error = None
            self._cond.notify_all()
        return float(times[-1]) if len(times) else 0.0

    def busy(self) -> bool:
        with self._cond:
            return bool(self._queue) or self._done_id != getattr(self, "_expect_id", 0)

    def wait(self, timeout: Optional[float] = None, tol: Optional[float] = None) -> bool:
        """等待当前轨迹结束（含到位检查）。返回是否到位。"""
        cid = getattr(self, "_expect_id", 0)
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._cond:
            while self._done_id != cid:
                remain = None if deadline is None else deadline - time.perf_counter()
                if remain is not None and remain <= 0:
                    self.log.warning("[%s] 等待轨迹超时（未收到完成）", self.arm)
                    return False
                self._cond.wait(0.05 if remain is None else min(remain, 0.05))
        if self.last_error is not None:
            return False
        if self._last_cmd_q is None:
            return True
        return self.check_arrival(self._last_cmd_q, tol=tol)

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """等待后台线程空闲（不发新指令）。"""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if not self.busy():
                return True
            time.sleep(0.01)
        return False

    def check_arrival(self, target: np.ndarray, tol: Optional[float] = None) -> bool:
        """读一次当前位置，判断是否到位；不会长时间阻塞。"""
        if tol is None:
            tol = float(self.cfg["motion"]["arrive_tol"])
        deadline = time.perf_counter() + float(self.cfg["motion"]["arrive_timeout_s"])
        best = None
        while True:
            try:
                q, _ = self.bus.read_state()
                err = float(np.abs(q - target).max())
                best = err if best is None else min(best, err)
                if err <= tol:
                    return True
            except BusError as exc:
                self.log.debug("[%s] 到位读取失败: %s", self.arm, exc)
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.04)
        if best is not None:
            self.log.warning("[%s] 到位残差 %.1f（容差 %.1f）", self.arm, best, tol)
        return False

    def move(self, waypoints: Sequence[Sequence[float]], **kwargs) -> bool:
        """阻塞式移动：start + wait。"""
        timeout = kwargs.pop("timeout", None)
        self.start(waypoints, **kwargs)
        return self.wait(timeout=timeout)

    def go(self, pose: Sequence[float], **kwargs) -> bool:
        return self.move([pose], **kwargs)

    def set_gripper(self, value: float, settle: Optional[float] = None) -> float:
        """只改夹爪，保持当前关节目标。"""
        q = self._planned_q if self._planned_q is not None else self.hold_q
        if q is None:
            q = self.bus.read_joints()
        v = float(clamp(value, 0.0, 100.0))
        self.bus.write_goal(q, v)
        self.hold_q = np.asarray(q, dtype=float).copy()
        self._planned_q = self.hold_q.copy()
        self.hold_gripper = v
        time.sleep(float(settle if settle is not None
                         else self.cfg["motion"]["gripper_close_settle_s"]))
        return self.read_gripper()

    def read_joints(self) -> np.ndarray:
        return self.bus.read_joints()

    def read_gripper(self) -> float:
        return self.bus.read_gripper()

    def read_load(self) -> float:
        return self.bus.read_load()

    def read_temp(self) -> float:
        return self.bus.read_temp()

    def open_gripper(self) -> float:
        return self.set_gripper(float(self.cfg["arms"][self.arm]["gripper_open"]),
                                settle=float(self.cfg["motion"]["gripper_open_settle_s"]))


class DualArm:
    """两条臂的集合，便于统一初始化/读取/停机。"""

    def __init__(self, cfg: dict, logger, stop_event: threading.Event):
        self.cfg = cfg
        self.log = logger
        self.stop_event = stop_event
        # 每条臂一份标定：官方 lerobot-calibrate 是分开标的，文件也是分开的。
        paths = calibration_paths(cfg)
        self.calibration_paths = paths
        self.calibrations: dict[str, dict[str, MotorCalibration]] = {}
        self.ports = resolve_arm_ports(cfg, logger)
        self.arms: dict[str, Arm] = {}
        for name in ("left", "right"):
            if name not in paths:
                raise SystemExit(
                    f"config.json 里没有 {name} 臂的电机标定路径"
                    f"（devices.{name}_calibration_json）")
            cal = load_arm_calibration(paths[name], name)
            self.calibrations[name] = cal
            bus = ArmBus(name, cfg, self.ports[name], cal, logger, stop_event,
                         calibration_path=str(paths[name]))
            self.arms[name] = Arm(name, cfg, bus, logger, stop_event)

    def connect(self, only: Optional[Sequence[str]] = None) -> None:
        """连接并上力。only 指定时只处理这些臂（标定单臂时用）。

        任一条臂彻底连不上就抛异常；但每条臂内部已经把“参数写失败/单电机
        上力失败”降级成警告，不会因为一个丢包就整条臂不可用。
        """
        names = list(only) if only else ["left", "right"]
        connected = []
        try:
            for name in names:
                self.arms[name].bus.connect()
                connected.append(name)
            for name in names:
                self.arms[name].bus.enable_torque()
        except Exception:
            # 连到一半失败：把已连上的臂卸力断开，别留下半通电状态
            for name in connected:
                try:
                    self.arms[name].bus.disable_torque()
                except Exception:
                    pass
                try:
                    self.arms[name].bus.disconnect()
                except Exception:
                    pass
            raise
        # 让每条臂记录当前位姿，避免第一条轨迹从零开始插值
        for name in names:
            arm = self.arms[name]
            for attempt in range(3):
                try:
                    q, g = arm.bus.read_state()
                    arm.hold_q = q
                    arm._planned_q = q.copy()
                    arm.hold_gripper = g
                    arm.bus.gripper_cmd = g
                    self.log.info("[%s] 初始位姿 %s 夹爪 %.1f", name,
                                  np.round(q, 1).tolist(), g)
                    break
                except Exception as exc:
                    if attempt == 2:
                        self.log.warning("[%s] 读初始位姿失败: %s", name, exc)

    def disconnect(self) -> None:
        for name in ("left", "right"):
            self.arms[name].bus.disconnect()

    def disable_torque(self) -> None:
        for name in ("left", "right"):
            self.arms[name].bus.disable_torque()

    def enable_torque(self) -> None:
        for name in ("left", "right"):
            self.arms[name].bus.enable_torque()

    def reconnect(self, name: Optional[str] = None) -> None:
        names = [name] if name else ["left", "right"]
        for n in names:
            try:
                self.arms[n].bus.connect()
                self.arms[n].bus.enable_torque()
                q, g = self.arms[n].bus.read_state()
                self.arms[n].hold_q = q
                self.arms[n]._planned_q = q.copy()
                self.arms[n].hold_gripper = g
            except Exception as exc:
                self.log.error("[%s] 重连失败: %s", n, exc)
                raise

    def max_temp(self) -> float:
        temps = [self.arms[n].read_temp() for n in ("left", "right")]
        return max(temps) if temps else 0.0

    def error_streak(self) -> int:
        return max(self.arms[n].bus.error_streak for n in ("left", "right"))

    def stop_workers(self) -> None:
        self.stop_event.set()
        for n in ("left", "right"):
            self.arms[n]._worker.join(timeout=1.0)

    # ---- 并行工具 ----
    def start_both(self, left_wps, right_wps, **kw) -> None:
        self.arms["left"].start(left_wps, **kw)
        self.arms["right"].start(right_wps, **kw)

    def wait_both(self, timeout: Optional[float] = None) -> bool:
        ok_l = self.arms["left"].wait(timeout=timeout)
        ok_r = self.arms["right"].wait(timeout=timeout)
        return ok_l and ok_r
