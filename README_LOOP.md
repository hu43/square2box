# square2box_loop — 24/7 抓取循环

`square2box_loop.py` 是在 `square2box.py` 基础上做的无人值守版本，完成：

```
桌面抓取 → 放入盒子 → 从盒子抓出 → 放回桌面 → 重复
```

核心改动：

- **YOLO 检测**：本地 YOLOE 开放词表分割（`yoloe-11l-seg.pt`），HSV 做精修和兜底；不再只靠颜色阈值。
- **平滑运动**：50Hz `smootherstep` 插值，运动过程中不读总线，到位后用反馈确认；视觉伺服按相机帧率跑。
- **抓取验证**：夹爪位置 + `Present_Load`/`Present_Current` + 抬升后视觉复检。
- **自动恢复**：相机看门狗、机械臂重连、连续失败硬恢复、电机温度保护。
- **可观测**：轮转日志 + 每轮 CSV 统计。

## 1. 环境

```bash
conda activate hupy3.12
cd /home/zhoulin/hu/square2box
```

确认 YOLO 资产（本机已有，脚本也会自动找）：

```bash
ls -l yoloe-11l-seg.pt mobileclip_blt.ts
# 如果没有，建软链接：
ln -sf /home/zhoulin/hu/lerobot/yoloe-11l-seg.pt yoloe-11l-seg.pt
ln -sf /home/zhoulin/hu/lerobot/examples/mobileclip_blt.ts mobileclip_blt.ts
```

设备符号链接（`/dev/left_arm`、`/dev/left_wrist_cam`）丢失时，脚本会自动回退到原始设备：

- 左臂（8 电机，含头）：`/dev/ttyACM1`
- 右臂（6 电机）：`/dev/ttyACM0`
- 左腕相机：`/dev/video2`

也可以在 `config.json` 显式写 `left_arm_port`、`right_arm_port`、`camera_device`。长期使用建议按当前 USB 拓扑更新 `/etc/udev/rules.d/99-xlerobot.rules` 并 `udevadm trigger`。

## 2. 先自检（不动电机）

```bash
python square2box_loop.py self-test
python square2box_loop.py preview --seconds 10
# 想直接看 hover 位下的检测效果（会动电机，慢速）：
python square2box_loop.py preview --hover --seconds 10
# 看盒子上方的检测/盒子位置（会动电机）：
python square2box_loop.py preview --box --seconds 10
```

`self-test` 会检查配置、相机、YOLO、机械臂只读状态；`preview` 只开相机和 YOLO；`preview --hover` 会先慢速移到 `hover_joints`。

如果 `preview` 看不到方块：

- 先把机械臂摆到 `hover_joints` 附近，让腕相机能看到桌面；
- 调 `config.json` 的 `yolo_conf`（默认 0.06）和 HSV 范围；
- 确认 `mobileclip_blt.ts` 软链接有效。

## 3. 标定

### 3.1 全量重新标定点位（推荐）

机械臂、相机、盒子位置变了，或者想从头重标，用：

```bash
cp config.json config.before_recalib.json   # 先备份
python calibrate_all.py --check   # 先只读检查相机/YOLO/当前配置，不动电机
python calibrate_all.py           # 交互式全量标定
python calibrate_all.py --no-probe # 只标点位，不自动探测增益
```

脚本会卸力，按提示用手摆位，依次记录：

| 键 | 含义 |
|---|---|
| `gripper_open` / `gripper_closed` | 夹爪完全张开 / 夹住方块 |
| `hover_joints` | 桌面上方，腕相机能看到方块 |
| `grasp_joints` | 桌面抓取深度（实际使用时会按 `grasp_height_offset_deg` 抬高） |
| `grasp_height_offset_deg` | 抓取高度补偿，正数=更高；撞桌面就加大，夹太高就减小 |
| `box_hover_joints` | 盒子上方 |
| `box_drop_joints` | 盒内释放高度 |
| `box_pick_joints` | 盒内抓取深度 |
| `table_drop_joints` | 桌面释放高度 |
| `box_look_joints` | 看盒子内部的观察位 |
| `home_joints` | 安全休息位 |
| `grasp_pixel` / `grasp_area` | hover 位检测到的方块中心/面积 |
| `grasp_depth_pixel` | 抬高后的抓取位处检测到的方块中心（`calibrate-depth` 会更新） |
| `box_target_pixel`（可选） | 盒内验证 ROI 中心 |

标定过程中会把带检测框的画面保存到 `calib_frames/`，方便检查。

点位录完后，脚本可选自动探测：

- `gain_matrix`：hover 位每个 `servo_joints` +3°，算像素增益；
- `depth_gain_matrix`：抓取深度 pan/lift/elbow/wrist ±2°，算 2×4 雅可比；
- `roll_slope_deg_per_deg`：腕部 +8°，算角度响应斜率。

自动探测会通电小幅移动，脚本会先提示确认手已离开。全部结果直接写回 `config.json`。

标定完成后：

```bash
python square2box_loop.py self-test
python square2box_loop.py run --cycles 1
```

### 3.2 只补新增位姿

如果桌面/盒子主位姿没变，只想补循环版本新增的四个位姿：

```bash
python square2box_loop.py calibrate-extra
```

按提示记录 `box_pick_joints`、`table_drop_joints`、`box_look_joints`、`home_joints`。

不跑也能用：`table_drop` 会从 `grasp_joints` 推导；如果 `table_drop_joints` 和 `grasp_joints` 相差超过 10 个单位（例如误录成折叠位），会被自动忽略。`box_pick`/`box_look` 现在会直接采用，但**盆内抓出建议用 3.4 的专用命令**，因为它会同时记录观察/抓取像素并允许单独抬高。

调过 `grasp_height_offset_deg` 之后，建议重跑一次深度标定（会动电机，约 1 分钟）：

```bash
python square2box_loop.py calibrate-depth
```

它会在“抬高后的抓取位”同时记录 `grasp_depth_pixel` 和 2×4 深度雅可比，写回 `config.json`。不跑也可以：第一次抓取时会在深度处现场探测雅可比，但 `grasp_depth_pixel` 仍是旧值，可能把夹爪从方块旁拉走。

### 3.3 重新标定夹爪开合

只改 `gripper_open` / `gripper_closed` 两个值，不影响其他位姿：

```bash
# 只读当前配置和夹爪读数，不动
python calibrate_gripper.py --read

# 卸力，手摆完全张开 / 合拢到能夹住方块，回车记录并保存
python calibrate_gripper.py

# 记录后通电验证开/合
python calibrate_gripper.py --test
```

标定时会卸力，先托住手臂；“合拢”是合到能夹住方块的程度，不要压到最死。循环运行时会在自由空间自动测一次空合值，所以这里只记录开合命令值。

### 3.4 单独标定盆内抓出

盆内抓出现在是独立策略，不再从“放入盆内”的 `box_drop` 推导，也不会用桌面的 `grasp_pixel`。单独录一对 `box_look` / `box_pick`：

```bash
# 卸力，手摆观察位和抓取位，回车记录
python square2box_loop.py calibrate-box-pick

# 记录后通电验证闭爪
python square2box_loop.py calibrate-box-pick --test
```

标定要点：

- `box_look`：夹爪在盒口上方，能稳定看清方块；程序会先在此位置重新对中。
- `box_pick`：夹爪从两侧包住方块、但不碰盆底的位置，宁可略高；方块应能在两指之间被看见。
- 抓取位画面会保存到 `logs/box_pick_calibration.jpg`。若看不到方块或检测面积超过 6000 px（通常意味着指爪/盆沿遮挡严重），标定会拒绝保存。
- 盒内抓取只使用专用 `box_look`/`box_pick` 位姿，不复用桌面深度雅可比微调；在盒内视角下，该雅可比方向和可达性未标定。
- 保存时会把 `box_pick_height_offset_deg` 重置为 0；如果之后还撞盆底，应重新手动摆高并标定，而不是继续向下探。

## 4. 试跑

先跑 1 轮，手放在急停/电源附近：

```bash
python square2box_loop.py run --cycles 1
```

确认完整走完“抓取→放入→抓出→放回”后，再跑无限循环：

```bash
python square2box_loop.py run --cycles 0
```

只开相机/YOLO、不动电机：

```bash
python square2box_loop.py run --dry-run
```

停止：`Ctrl+C`。脚本会先开爪、再下电力矩。

### 实测结果（2026-10-29，本机 SO-101 左臂）

- **5 分钟连续：10 轮成功 / 0 失败**，单轮 28–41 秒；中途 73°C 触发一次卸力降温，30 秒后自动恢复并继续。
- **第二次 5 分钟：5 轮成功 / 1 轮失败后自动恢复**；一次 45 秒降温。
- 温度是 24/7 的主要限制：如果频繁触发，建议加风扇、降低 `max_vel_deg_s`（如 35）或增大 `cycle_pause_s`。
- 抓取验证是“位置 + 负载 + 抬升后视觉复检”；偶尔会因视觉误判多试一次，但不会把空爪当成成功。

### 实测结果（2026-10-30，重新标定 + 抬高抓取高度后）

- 2 轮连续成功：67.3s、59.2s，桌面/盒内抓取均一次夹住。
- 5 分钟运行：**3 轮成功（65s / 65s / 91s）+ 1 轮失败后自动恢复**；失败原因是盒内抓取偏弱、方块留在盒里，恢复流程把它取回桌面。温度 79°C 触发一次 1/2 警告。
- 抓取高度补偿 `grasp_height_offset_deg=6` 解决了夹爪撞桌面；盒内抓取不再套用桌面的 `grasp_depth_pixel`，改用标定好的 `box_pick` 位姿，并在抬升后复检位置/负载，滑脱会判失败。
- 如果盒内抓取仍偏弱，先确认盒子深度/假底是否让夹爪够得到；再调 `box_pick_deepen_deg`（2→4）或重录 `box_pick_joints`。

> `conda run` 默认会缓冲输出，长时间循环看不到实时日志；用 `conda run --no-capture-output -n hupy3.12 python square2box_loop.py run ...`，或先 `conda activate hupy3.12` 再 `python -u square2box_loop.py run ...`。

## 5. 24/7 常驻（systemd）

```bash
mkdir -p logs
sudo cp square2box-loop.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now square2box-loop
sudo systemctl status square2box-loop
```

查看日志：

```bash
tail -f logs/square2box_loop.log
column -s, -t logs/cycles.csv | tail
```

服务配置了 `Restart=always`，进程崩溃/退出后 5 秒自动拉起。

## 6. 关键参数（config.json）

| 参数 | 默认 | 说明 |
|---|---|---|
| `yolo_model` | `yoloe-11l-seg.pt` | 本地 YOLOE 模型 |
| `yolo_prompts` | `["green cube","green block","cube"]` | 开放词表提示词，按方块颜色改 |
| `yolo_conf` | 0.06 | 低一点保召回，HSV 负责过滤 |
| `max_vel_deg_s` | 45 | 关节最大速度（归一化单位/秒） |
| `control_hz` | 50 | 轨迹发送频率 |
| `servo_kp` | 0.7 | 视觉伺服比例增益 |
| `servo_step_deg` | 2.0 | 单次伺服关节增量上限 |
| `servo_tol_px` | 6 | 悬停对中像素容差 |
| `depth_tol_px` | 12 | 深度对准像素容差 |
| `grasp_height_offset_deg` | 6.0 | 抓取高度补偿；撞桌面就加大，夹太高就减小 |
| `grasp_retries` | 3 | 抓取重试次数 |
| `grasp_pos_margin` | 0.6 | 夹爪读数高于空合值多少算夹住（软方块用小值） |
| `grasp_load_min` | 20.0 | 负载阈值；空载异常时自动禁用 |
| `box_target_pixel` | `[320,240]` | 盒内验证 ROI 中心 |
| `box_roi_radius` | 150 | 盒内验证 ROI 半径 |
| `box_pick_deepen_deg` | 2.0 | 没有单独 `box_pick_joints` 时的兜底下探角度 |
| `box_pick_height_offset_deg` | 0.0 | 盆内抓取整体抬高量；撞盆底就加 2–5，正数=更高 |
| `box_look_pixel` | None | 盆内观察位对中像素；`calibrate-box-pick` 自动记录 |
| `box_pick_pixel` | None | 盆内抓取位微调像素；`calibrate-box-pick` 自动记录 |
| `temp_limit_c` | 58 | 电机温度保护阈值 |
| `temp_wait_s` | 45 | 温度保护等待时间 |
| `max_consecutive_failures` | 5 | 连续失败几次后硬恢复 |

## 7. 现场建议

- **盒子**：开口尽量大（≥12cm）、壁低、内壁光滑；方块能稳定落在底部，夹爪能伸进去。透明盒/高壁盒会让视觉验证和盒内抓取变难；深盒建议垫 3–5cm 假底。
- **颜色**：方块和盒子不要同色；YOLOE 提示词按方块颜色改（如 `"red cube"`），HSV 也改成对应范围。
- **光照**：避免强反光和频闪；YOLOE 比 HSV 抗光照，但 HSV 精修仍依赖颜色。
- **线缆**：循环运动范围比单次抓取大，确认腕相机线、夹爪线不会被扫到。
- **温度**：24/7 运行时每 5 秒检查一次温度；超过 60°C 自动卸力等待 30 秒。如果频繁触发，降低 `max_vel_deg_s` 或增加 `cycle_pause_s`。
- **急停**：首次试跑务必有人在场；确认 `Ctrl+C` 能在 1–2 秒内停住。

## 8. 常见问题

**YOLO 检测不到**
- 跑 `preview` 看实时输出；确认方块在画面里、颜色/光照正常。
- 降低 `yolo_conf` 到 0.03；或把 `yolo_prompts` 改成更贴近的英文描述。
- 确认 `mobileclip_blt.ts` 不是 0 字节/损坏文件。

**总是空爪**
- 跑 `calibrate-extra` 重新录 `grasp_joints`/`box_pick_joints`。
- 调 `grasp_pos_margin`（默认 0.6；软方块可降到 0.4，靠负载/视觉复检兜底）。
- 检查 `grasp_depth_pixel` 是否对准指间；`preview` 里看红色十字。
- 如果启动日志里“空合值”一直偏高（>20），说明测量位夹爪被挡：检查 `measure_pan_offset`（默认 hover 横向 +20°、抬高 10°），或把 `safe_joints` 设到完全自由的姿态。

**放入盒子失败**
- 检查 `box_drop_joints` 是否在盒子内部、足够低。
- 盒子太深/太窄会导致夹爪撞壁；换宽口浅盒。
- 调 `box_target_pixel` / `box_roi_radius`，让盒内验证 ROI 对准盒子。

**盒内抓不出**
- 如果日志出现“盒内抬升后 ... 滑脱”，说明闭爪瞬间碰到了方块/盒壁，但抬起来时方块滑回盒里。常见原因是**盒子太深、方块落在底部，夹爪够不到**。
- 最有效的办法：在盒底垫一块 **3–5cm 厚的泡沫/木块做假底**，把方块抬到夹爪能套住的高度；或直接换**浅口托盘/低壁宽口盒**。
- 盒子变浅/加假底后，重跑 `calibrate-extra` 重录 `box_pick_joints`/`box_look_joints`。
- `box_pick_deepen_deg` 只有在盒子还有余量时才加；如果加到 10 以上开始压盒底，就不要继续加。
- 方块贴壁时，先把方块拨到盒子中间；必要时用胶带/挡块限制方块落点。

**电机过热/堵转**
- 脚本会在 `temp_limit_c`（默认 58°C）卸力等待 `temp_wait_s`（默认 45s），然后自动重连继续。
- 频繁触发时：加风扇、降低 `max_vel_deg_s`（如 35）、增加 `cycle_pause_s`。
- 检查夹爪闭合值是否压得太死；`gripper_closed` 不要比空合值低太多。

## 9. 文件

- `square2box_loop.py` — 主程序
- `calibrate_all.py` — 交互式全量点位标定
- `config.json` — 标定和参数（脚本会合并新键，不会删旧键）
- `calib_frames/` — 标定时保存的带检测框画面
- `logs/square2box_loop.log` — 轮转日志
- `logs/cycles.csv` — 每轮成功/失败/耗时
- `square2box-loop.service` — systemd 常驻配置
