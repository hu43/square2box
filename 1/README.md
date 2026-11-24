# 1/ — 标定后无摄像头双臂巡演

在 `2arm/calibrate2arm.py` 标定完成后，用本目录的 `run_calibrated_flow.py`
让两条 SO-101 按刚标定好的 `home/carry/桌面/盒内` 点位跑一条固定动作流
（`measure` 只用于标定时的空夹基线，不参与巡演）。**播放过程不打开摄像头，
也不导入 `armvision` / `cv2` / `ultralytics`。**

## 四个入口

| 入口 | 用途 |
|---|---|
| `./run_tour.sh` | 标定点巡演：只走 9 个标定位姿，模拟抓放，不要求桌面有方块 |
| `./run_handoff.sh` | 恢复型手递循环：带抓空重试和四点搜索，适合长时间无人值守 |
| `./run_handoff_flow.sh` | **10 步示教 flow 原样回放**：严格按示教好的位置和动作顺序执行 |
| `./run_taught_flow.sh` | **14 步预设动作流原样回放**：左臂 7 步 + 右臂 7 步，严格按示教值执行 |

10 步示教 flow 由 `teach_handoff_flow.py` 生成/示教，详见“10 步示教动作流”。

14 步预设动作流由 `calibrate_taught_flow.py` 逐点示教，运行时只发
`taught_flow.json` 里的 14 个关节值；不叠加偏移、不派生 hover、不做
自适应恢复。详见“14 步预设动作流”一节。

`run_handoff.sh` 需要先做一次交叉位示教（`calibrate_handoff.py`），
详见“双臂循环手递”一节。

## 动作流

每条臂依次执行：

1. 张爪：`home -> carry -> table_approach -> table_hover -> table_pick`
2. 在 `table_pick` 合爪（模拟抓起），停留
3. 保持合爪：`table_pick -> table_hover -> carry -> box_approach -> box_hover -> box_pick -> box_drop`
4. 在 `box_drop` 开爪（模拟放下），停留
5. 张爪：`box_drop -> box_hover -> box_approach -> carry -> home`

默认左臂跑完再跑右臂；两臂不会同时穿越中间区域。使用 `--no-grasp`
可以全程张爪，只验证关节路径。

## 前置条件

- Conda 环境 `hupy3.12`（`numpy`、本机 LeRobot 可用）。
- 两条臂都已插好：`DualArm` 初始化会解析两个串口，即使只跑一条臂也需要
  两条臂的串口和标定文件都在。
- 工作区已清理，手离开机械臂。
- 启动时两臂最好停在 `home` 附近。脚本的第一段会直接走 `home`，不会先
  竖直抬离；如果上次停在桌面或盒内，回 `home` 的路径可能扫到物体，先用
  `--speed 0.3` 并观察，必要时手动把臂摆到空旷位置。

## 标定与同步配置

本目录是独立副本，不会自动读取 `2arm/config.json` 的后续改动。标定流程：

```bash
cd /home/zhoulin/hu/square2box/2arm
conda activate hupy3.12
python calibrate2arm.py                 # 或 --arm left / --arm right

# 标定完成后同步到本目录
cp /home/zhoulin/hu/square2box/2arm/config.json \
   /home/zhoulin/hu/square2box/1/config.json
```

以后每次重新标定（尤其是重新跑过 `lerobot-calibrate` 电机行程）之后，
都必须重新执行 `calibrate2arm.py`，再执行上面的 `cp`。只重标电机行程而
不重录点位时，旧点位会落在完全不同的物理角度，脚本会在上力前拒绝运行。

重标电机行程后请跑**完整**的 `python calibrate2arm.py`（不要用 `--only`），
因为 `--only` 会跳过夹爪开合值（`gripper_open/closed`）的记录；这些值同样
是按旧归一化坐标系录的。

> 本目录里的 `config.json` 是创建时从 `2arm` 复制的快照。如果 `./run_tour.sh
> --check` 报告“这些点位是在旧电机标定下录的”，说明快照已过期，按上面的
> 流程重新标定并同步即可。

## 运行

```bash
cd /home/zhoulin/hu/square2box/1

# 1) 只看动作序列，不连硬件
./run_tour.sh --dry-run

# 2) 只读校验：位姿完整性 + 电机固件标定指纹（不使能力矩）
./run_tour.sh --check

# 3) 首次实跑：先单臂、慢速
./run_tour.sh --arm left --speed 0.4

# 4) 双臂各一轮（默认）
./run_tour.sh

# 循环 / 定时
./run_tour.sh --cycles 3
./run_tour.sh --minutes 5
```

`run_tour.sh` 直接使用
`/home/zhoulin/miniconda3/envs/hupy3.12/bin/python`；也可以手动：

```bash
conda run -n hupy3.12 python run_calibrated_flow.py
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--arm` | `both` | `left` / `right` / `both` |
| `--cycles` | 0 | 巡演轮数；0 且没给 `--minutes` 时按 1 轮 |
| `--minutes` | 0 | 持续运行分钟数；0 表示不按时间限制 |
| `--speed` | 0.6 | 速度倍率（0.05~2.0），首跑建议 0.4 |
| `--step-pause` | 0.3 | 每段动作结束后的停留秒数 |
| `--grip-dwell` | 0.5 | 合爪/开爪后的停留秒数 |
| `--cycle-pause` | 0.3 | 两轮之间的停留秒数 |
| `--no-grasp` | 关 | 全程张爪，只走位姿 |
| `--dry-run` | 关 | 只打印计划，不连接硬件 |
| `--check` | 关 | 只读校验，不使能力矩 |
| `--home-only` | 关 | 选中的臂开爪回 `home` 后退出 |
| `--config` | `1/config.json` | 指定其他标定配置 |

## 安全与退出行为

- 启动前会检查 10 个位姿是否齐全、`gripper_open/closed` 是否存在，并拒绝
  含 ±100 饱和点位的配置（饱和点不可用）。
- **正式运行（包括 `--home-only`）在使能力矩之前会只读电机固件指纹**；
  如果 `config.json` 里的点位是在旧电机标定下录的，脚本会直接拒绝运动，
  不会先动一下再报错。
- 某一分段未到位（超时或残差超过 `motion.arrive_tol`）会立即中止本轮，
  不会继续走向下一个点位。
- 每轮开始读取电机温度：超过 `safety.temp_slow_c` 自动降速，达到
  `safety.temp_stop_c` 中止。
- **Ctrl+C**：停止轨迹线程 → 按当前关节位置开爪 → 卸力 → 断开。机械臂
  停留在当前姿态，不会自动回 `home`；确认现场安全后再手动回位或重跑。
- 正常结束：两臂已回 `home` 且张爪，随后卸力断开。
- 日志：`1/logs/calibrated_flow.log`，同时输出到终端。

## 常见错误

| 现象 | 原因 / 处理 |
|---|---|
| `缺少位姿 ...` | 标定未完成；重跑 `2arm/calibrate2arm.py --arm <arm>` 后同步 `config.json` |
| `这些点位是在旧电机标定下录的` | 重标过电机行程；必须重新标点位并同步 `config.json` |
| `只找到 1 条可用机械臂串口` | 两条臂的 USB/电源没都连上；修好后再运行 |
| `找不到电机标定文件` | `config.json` 里 `devices.*_calibration_json` 路径失效；检查官方 lerobot 标定文件 |
| 某分段 `未到位` | 先降低 `--speed`；仍失败则重标对应点位或检查机械阻力 |

---

# 双臂循环手递（无摄像头）

`dualarm_handoff_loop.py` 让两条臂在各自的 `table_pick` 前方来回传递同一个方块：

```
left_front（左臂前方）              right_front（右臂前方）
       │                                  │
       │ 1. left 抓起                     │
       └───────── 2. left 放到 ──────────>│
                                          │ 3. right 抓起
       │<──────── 4. right 放到 ──────────┘
       │
    方块回到 left_front，计 1 轮，继续循环
```

## 为什么需要一次交叉位示教

“左臂放到右臂前”必须用**左臂自己的关节位姿**到达右臂 `table_pick` 的
物理位置。不同臂的关节坐标不能互换，不能把右臂的 `table_pick` 直接发给
左臂。`calibrate_handoff.py` 记录两个交叉抓取位：

- `left.place_right`：左臂在右臂 `table_pick` 位置、对方抓取高度
- `right.place_left`：右臂在左臂 `table_pick` 位置、对方抓取高度

这两个位姿同时用于“放”和“抓”：同一条臂在对方交接点抓取时也用它。
运行主程序时会**原封不动**地发送这些示教位姿，不会叠加释放抬高、
pan/深度偏移，也不会用别的点反推 hover。重试也只是重复同一个示教位姿。

## 首次准备

1. 电机标定/点位标定必须是最新的。当前 `config.json` 若是旧快照，先：
   ```bash
   cd /home/zhoulin/hu/square2box/2arm
   python calibrate2arm.py            # 完整模式，不要 --only
   cp config.json ../1/config.json
   ```
   **重标过电机行程后，两个交叉位也必须重新示教**（它们的归一化坐标
   同样会变）；`--check` 会把旧指纹拦下来。
2. 示教两个交叉位：
   ```bash
   cd /home/zhoulin/hu/square2box/1
   python calibrate_handoff.py        # 依次示教 left.place_right / right.place_left
   ```
   程序会先把参考臂移到 `table_hover` 让你看清交接点，然后卸力，用手把
   另一条臂的夹爪摆到方块上（两指包住、与对方抓取高度一致），回车记录。
3. 校验：
   ```bash
   ./run_handoff.sh --check
   ./run_handoff.sh --dry-run
   ```

## 运行

运行前：

- 把方块放在 **left_front**（左臂 `table_pick` 的位置，左臂正前方）。
- 两臂最好在 `home` 附近；不确定时先 `./run_handoff.sh --home-only`。

```bash
./run_handoff.sh --minutes 5 --speed 0.7   # 先跑 5 分钟验证
./run_handoff.sh --cycles 10 --speed 0.8
./run_handoff.sh                            # 无限循环（24h 用这个）
```

如果方块不在左臂前（例如重启后方块停在右臂前），脚本会先在左前抓取失败，
然后自动做四点搜索找到它并继续；这会多花约二三十秒，但不需要人工搬回。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--minutes` | 0 | 运行分钟数；0=不限时 |
| `--cycles` | 0 | 完整轮数（方块回到左臂前）；0=不限 |
| `--speed` | 1.0 | 全局速度倍率；首跑建议 0.5~0.7 |
| `--pick-tries` | 3 | 每个交接点重复同一示教位的次数（不做偏移） |
| `--no-overlap` | 关 | 放置臂先回 home，再让下一条臂预位 |
| `--stall-restart` | 300 | 连续无进展多少秒退出等待重启；0=禁用 |
| `--csv` | `logs/handoff.csv` | 每轮/失败记录 |
| `--dry-run` | 关 | 只打印动作流，不连硬件 |
| `--check` | 关 | 只读校验位姿/指纹 |
| `--home-only` | 关 | 两臂回 home 后退出 |

## 可靠性 / 恢复

- 每次合爪都做“空夹基线 + 位置 + 负载”双重判断；抬升后再复检一次，
  滑脱会重复同一个示教点，不改变位置。
- 抓取失败先在同一交接点换另一条臂重试（它也有一组示教位姿）；两边都
  失败才判定方块位置未知。
- 方块位置未知时，两条臂分别在左右两个交接点做四点搜索；找到后按当前
  位置继续，不整轮重来。
- 温度超过 `temp_slow_c` 自动降速；达到 `temp_stop_c` 时若手里没有方块，
  卸力冷却后回 home 并重测空夹基线；手里有方块时先完成这次放置。
- 总线连续错误会重连；连续 `--stall-restart` 秒无进展会退出，交给
  systemd 重启。

## 24 小时运行

安装 systemd 服务（按本机用户名/组修改）：

```bash
sudo cp dualarm-handoff.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dualarm-handoff.service
journalctl -u dualarm-handoff.service -f
```

服务使用 `Restart=always`、`RestartSec=5`、`KillSignal=SIGINT`。停止：
`sudo systemctl stop dualarm-handoff.service`（会走脚本的“开爪→卸力”清理）。

## 已知边界

这是**纯盲操作**：没有摄像头确认方块落点。如果方块被碰落到桌面外，或
两个交接点在物理上不可达，四点搜索也找不到它，循环会持续重试并最终由
`--stall-restart` 退出重启。要真正做到长期无人值守，需要：

- 方块始终在两条臂可达的台面范围内；
- 两个交叉位都在两臂可达范围内（`--check` 会拦饱和/贴边）；
- 夹爪负载判断有效（若夹爪不返回负载，只靠位置判断，可靠性会下降）。

---

# 10 步示教动作流（预设 10 个动作）

这套流程把“每条臂的 10 个动作”直接存成 JSON，运行时逐条按示教值发送。
它不依赖状态机、不做四点搜索，**位置和动作顺序和示教时完全一致**。

## 预设的 10 步

左臂（右臂把 `place_right` 换成 `place_left`）：

| # | 动作 | 移动时夹爪 | 到位后 |
|---|---|---|---|
| 1 | `home` | open | - |
| 2 | `carry` | open | - |
| 3 | `table_approach` | open | - |
| 4 | `table_hover` | open | - |
| 5 | `table_pick` | open | **合爪** |
| 6 | `table_hover` | closed | - |
| 7 | `carry` | closed | - |
| 8 | `place_right` / `place_left` | closed | **开爪** |
| 9 | `carry` | open | - |
| 10 | `home` | open | - |

每步的 `q` 是示教值，原样发送；`dwell` 只是停留时间，不改变位置。

## 生成 / 示教

flow 文件在 `flows/`：

- `flows/left_handoff_10.json`
- `flows/right_handoff_10.json`

用当前 `config.json` 生成预设：

```bash
cd /home/zhoulin/hu/square2box/1
python teach_handoff_flow.py --seed
```

逐点微调（程序先移到该步参考位，再卸力让你微调）：

```bash
python teach_handoff_flow.py                 # 两条臂全部 10 步
python teach_handoff_flow.py --arm left      # 只左臂
python teach_handoff_flow.py --step 8 --arm left   # 只重教“左臂到右臂前”
python teach_handoff_flow.py --step 8 --arm right  # 只重教“右臂到左臂前”
```

操作：回车记录，`t` 加力保持/卸力，`s` 保留当前参考值，`q` 退出。
示教只改关节角度；10 步顺序和夹爪动作是预设的。

## 原样运行

```bash
./run_handoff_flow.sh --check
./run_handoff_flow.sh --dry-run
./run_handoff_flow.sh --minutes 5 --speed 0.7
./run_handoff_flow.sh                     # 无限循环
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--minutes` | 0 | 分钟数；0=不限时 |
| `--cycles` | 0 | 完整轮次（左 flow + 右 flow 都成功）；0=不限 |
| `--speed` | 1.0 | 全局速度倍率 |
| `--flow-tries` | 2 | 单条 flow 失败后原样重试次数 |
| `--max-failures` | 3 | 连续失败多少次后退出等待重启；0=一直重试 |
| `--cycle-pause` | 0.15 | 一轮完成后的停留秒数 |
| `--dry-run` | 关 | 只打印 10 步，不连硬件 |
| `--check` | 关 | 只读校验 q/指纹 |
| `--home-only` | 关 | 两臂回 home 后退出 |

运行逻辑：

1. 左臂跑完整 10 步 flow（结束在 home）；
2. 右臂跑完整 10 步 flow（结束在 home）；
3. 算一轮，重复。

每条 flow 的目标只有 JSON 里的 q；重试也是原样重跑同一条 flow，
不会换 pan/深度、不会抬高释放点、不会用别的点反推 hover。

如果某一步执行失败且夹爪仍夹着方块，脚本**不会自动回 home**（避免边走边
开爪把方块甩掉），而是停在当前位置等待人工处理或 systemd 重启。正常完成
的 flow 最后一步是 `home` + 张爪，所以两条 flow 之间两臂都在安全位。

## 24 小时服务

```bash
sudo cp handoff-flow.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now handoff-flow.service
journalctl -u handoff-flow.service -f
```

`Restart=always`、`RestartSec=5`、`KillSignal=SIGINT`。

## 和 `run_handoff.sh` 的区别

| | `run_handoff.sh` | `run_handoff_flow.sh` |
|---|---|---|
| 位置来源 | 10 个标定点 + 2 个交叉点 | 示教好的 10 步 flow JSON |
| 动作顺序 | 代码内固定，失败时按状态机恢复 | 严格按 10 步 flow，失败只原样重跑 |
| 抓空/滑脱 | 换臂重试、四点搜索 | 重跑同一条 flow，不改变位置 |
| 适用 | 长时间无人值守、允许恢复动作 | 要求位置/顺序和示教完全一致 |

如果现场要求“死编程、完全按示教跑”，用 `run_handoff_flow.sh`；
如果需要长期无人值守自动找方块、抓空后换臂/搜索，用 `run_handoff.sh`。
严格 flow 运行不做夹爪载荷判定，抓取是否成功只看动作是否到位；对可靠性
要求更高时用恢复型循环。

---

# 14 步预设动作流（严格示教回放）

`calibrate_taught_flow.py` 按固定 14 步示教；`run_taught_flow.py` 按同一
顺序原样发送示教关节值。它和 10 步 flow 的区别：14 步把每条臂的
安全位 / 夹方块前 / 夹方块 / 中转位 / 放方块前 / 放方块 / 回安全位
全部单列。

## 预设的 14 步

| # | 动作 | 臂 | 移动时夹爪 | 到位后 |
|---|---|---|---|---|
| 1 | 左臂安全位 | left | open | - |
| 2 | 左臂夹方块前 | left | open | - |
| 3 | 左臂夹方块 | left | open | **合爪** |
| 4 | 左臂夹起方块并在中转位 | left | closed | - |
| 5 | 左臂放方块前 | left | closed | - |
| 6 | 左臂放方块 | left | closed | **开爪** |
| 7 | 左臂回安全位 | left | open | - |
| 8 | 右臂安全位 | right | open | - |
| 9 | 右臂夹方块前 | right | open | - |
| 10 | 右臂夹方块 | right | open | **合爪** |
| 11 | 右臂夹起方块并在中转位 | right | closed | - |
| 12 | 右臂放方块前 | right | closed | - |
| 13 | 右臂放方块 | right | closed | **开爪** |
| 14 | 右臂回安全位 | right | open | - |

## 方块流转（示教时按这个摆）

1. 初始把方块放在**左臂夹方块(3)** 的抓取点。
2. 第 6 步「左臂放方块」记录后自动开爪，方块落在**左臂放方块**的位置。
3. 第 10 步「右臂夹方块」必须把右臂摆到**同一个物理位置**，记录后自动合爪。
4. 第 13 步「右臂放方块」记录后自动开爪，方块落在**右臂放方块**的位置。
5. 下一轮第 3 步「左臂夹方块」必须示教到**同一个物理位置**。

也就是说：**第 6 步 = 第 10 步的位置；第 13 步 = 下一轮第 3 步的位置。**

## 示教

```bash
cd /home/zhoulin/hu/square2box/1
python calibrate_taught_flow.py              # 14 步全部
python calibrate_taught_flow.py --arm left   # 只左臂 7 步
python calibrate_taught_flow.py --step 4     # 只重录第 4 步
python calibrate_taught_flow.py --check      # 只读校验
```

示教时：

- 程序按预设自动把夹爪设成该步的 pre 状态（张开/合上），并**一直保持
  夹爪力矩**；
- 5 个关节电机可以按 `t` 卸力，用手摆臂；夹爪仍夹住方块，方便示教中转位；
- 第 3/10 步记录后自动合爪；第 6/13 步记录后自动开爪；
- 回车记录，`t` 加力/卸力，`o`/`c` 手动开合夹爪，`s` 跳过，`q` 退出。

示教文件：`taught_flow.json`。14 步顺序和夹爪动作是预设的，示教只改
每步的 5 个关节值。

## 原样运行

```bash
./run_taught_flow.sh --check
./run_taught_flow.sh --dry-run
./run_taught_flow.sh --minutes 5 --speed 0.7
./run_taught_flow.sh                     # 无限循环
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--minutes` | 0 | 分钟数；0=不限时 |
| `--cycles` | 0 | 完整 14 步轮数；0=不限 |
| `--speed` | 1.0 | 全局速度倍率；首跑建议 0.5~0.7 |
| `--tries` | 1 | 每步失败后重复同一示教位的次数；1=严格一遍 |
| `--no-verify-grasp` | 关 | 关闭合爪负载检查（只按位置/顺序） |
| `--grip-dwell` | 0.2 | 合爪/开爪后额外停留秒数 |
| `--cycle-pause` | 0.1 | 一轮结束后的停留秒数 |
| `--stall-restart` | 300 | 连续无成功轮次多久退出等待重启；0=禁用 |
| `--no-precondition` | 关 | 启动时不预置到安全位（要求两臂已在安全位） |
| `--dry-run` | 关 | 只打印 14 步，不连硬件 |
| `--check` | 关 | 只读校验 q/指纹 |

运行逻辑：

1. 按索引 1..14 顺序执行，每一步的目标就是 `taught_flow.json` 里的 q；
2. 夹爪只在第 3/10 步合、第 6/13 步开；
3. 14 步完成算一轮，然后从第 1 步重复。

严格性：

- 不叠加释放抬高、pan/深度偏移、反推 hover；
- `--tries > 1` 也只是重复同一步骤的同一个示教点；
- 不做换臂重试、四点搜索等自适应恢复；
- 某步失败会记录并继续按顺序执行；要严格一遍就保持 `--tries 1`。

启动时程序会先把两臂预置到第 1/8 步（各自安全位），之后才开始 14 步循环。
这是启动安全预置，不属于循环动作。

## 24 小时服务

```bash
sudo cp taught-flow.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now taught-flow.service
journalctl -u taught-flow.service -f
```

`Restart=always`、`RestartSec=5`、`KillSignal=SIGINT`。

## 与 10 步 flow 的区别

| | `run_handoff_flow.sh`（10 步） | `run_taught_flow.sh`（14 步） |
|---|---|---|
| 每条臂步数 | 5 步 | 7 步（多了安全位、放方块、回安全位） |
| 位置来源 | `flows/*.json` | `taught_flow.json` |
| 夹爪动作 | 预设 | 预设，按 14 步表 |
| 失败处理 | 原样重跑整条 flow | 按步重试；不做自适应恢复 |
| 适用 | 10 步够用的现场 | 明确要求 14 个动作点/顺序的现场 |
