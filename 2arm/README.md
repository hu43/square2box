# 双臂方块循环夹放（2arm）

左臂从桌面夹绿色方块放进盒子 → 右臂从盒里夹出来放到桌面 → 右臂从桌面夹起来放回盒里 → 左臂从盒里夹出来放回桌面，**一轮结束回到起点，可以无人值守循环往复**。

```
2arm/
├── armcore.py          公共层：配置、位姿、轨迹规划、日志
├── armdriver.py        双臂驱动：两个串口独立后台线程、100Hz 轨迹流
├── armvision.py        双腕相机、YOLOE 分割 + HSV 兜底、小幅对准
├── calibrate2arm.py    ★ 点位标定（先跑完这个才能调试动作）
├── four_step_cycle.py  ★ 推荐入口：四步循环 + 循环往复 + 每轮自恢复
├── dualarm_cycle.py    备选：视觉伺服版循环（自学习 hover/pick 像素）
├── dualarm_loop.py     旧版主循环 + 看门狗 + systemd 入口（保留作回退）
├── selftest_offline.py 无硬件离线验证（轨迹/状态机/恢复）
├── config.json         配置（点位先为 null，标定后自动写入）
└── dualarm-loop.service systemd 常驻服务
```

---

## ★ 推荐入口：`four_step_cycle.py`（四步循环，可往复）

一轮四个动作，最后方块回到左臂的桌面抓取点，下一轮直接从第 1 步接着来：

```
1. 左臂：桌面 -> 盒内
2. 右臂：盒内 -> 桌面
3. 右臂：桌面 -> 盒内
4. 左臂：盒内 -> 桌面   （回到起点）
```

```bash
conda activate hupy3.12
cd /home/zhoulin/hu/square2box/2arm

python four_step_cycle.py                    # 单轮，人守着急停
python four_step_cycle.py --minutes 5        # 连续跑 5 分钟（到点自动正常停止）
python four_step_cycle.py --cycles 10        # 跑 10 轮
python four_step_cycle.py --start-box        # 方块已在盒内，跳过第 1 步
python four_step_cycle.py --stop-after 2     # 只跑到第 2 步（调试用）
python four_step_cycle.py --keep-in-hand     # 收尾不放下，方块留在左爪
```

`--minutes` / `--cycles` 同时给时谁先到就按谁停；都不给只跑一轮。

### 它和另外两个入口的区别

| 环节 | `four_step_cycle.py` 的做法 |
|---|---|
| 右臂盒内抓取 | 盒口**二维视觉对准**：`pan` 管横向、`elbow` 管纵深，增益每次**现场实测**（约 −3.0 / −5.5 px/°），再按腕转 0/±15/±45° + 加深三轮扫描 |
| 夹不稳 | 只搭住方块顶棱时**原地松爪换角度，绝不抬升**——抬升会把方块拖到盒子另一侧，之后 pan 撞限位就够不到了 |
| 桌面抓取 | 先在观察位把 `pan` 对到抓取点，再**纯竖直**下探（原先是斜着从 hover 走到 pick，会扫飞方块）；深度补回 `table_pick_height_offset` |
| 放到桌面 | 落点对齐该臂自己的 `table_pick`，**分步慢开爪**，放完**原地保持**；下一步直接**原地合爪**再夹起，不重新下探（重新下探常差几毫米夹空） |
| 失败处理 | 每轮独立 try/except：失败就 `recover_to_start`（抬离 → 回 home → 右臂桌面点试探抓 → 放回盒子 → 左臂取到桌面点），然后**继续下一轮**，进程不退出 |
| 库里的视觉兜底 | `right_arm_cycle` 的旧像素目标/雅可比只作最后兜底，并且捕获 `PhaseError`，不会因为一次 precision 动作超时把整轮打挂 |
| 停顿 | 盒内对准只做一次（不再每轮重测）、桌面免二次下探；实测单轮 110~150 秒 |

### 一轮里能看到的关键日志

```
[left]  桌面对准下探（pan +0.0）-> [...]
[right] 实测像素增益: pan -3.16px/°, elbow -5.69px/°
[right] 盒口对准第 2 次: c=(337,185) 误差=(+37,-15)
[right] 闭爪: pos=5.17 (空夹 2.94) load=304.0 -> 夹住
[right] 保持原位，等下一步原地合爪
[right] 桌面抓取：原地合爪（上一步刚放下）
# 第 1 轮完成，用时 147.1s
[轮次] 第 1 轮成功（成功 1 / 已跑 1，用时 147s）
```

失败时是 `[轮次] 第 k 轮失败: ...`，随后是 `恢复：...`，然后继续下一轮。
全部结束后打印 `[轮次] 循环结束：成功 x 轮 / 共 y 轮，总用时 z s`。

### 想提高成功率，优先看这三处

1. **左臂 `box_drop` 与右臂 `box_hover/box_pick` 要落在同一个点上**。
   左臂把方块放下后，右臂的腕相机必须能看见、手必须够得到；偏差超过一个
   方块时右臂会报“盒内看不到方块/没夹住”。重标左臂 `box_drop` 时，
   让方块落在右臂 `box_pick` 的正上方最稳。
2. **右臂 `table_hover/table_pick` 别压在盒子边沿、也别贴着自己的腕相机模块**，
   往桌面空处挪几厘米重标。方块落在盒子边沿会滚进盒里，第 3 步就抓空。
3. **左腕相机 `/dev/video2` 要接稳**。它一掉线，左臂就完全无视觉，
   第 1、4 步只能靠 pan 扫描，单轮会多花 30~60 秒。

---

## 备选入口：`dualarm_cycle.py`（YOLO 视觉伺服版）

旧版 `dualarm_loop.py` 保留作回退。这个程序针对“盒内抓取不稳、下探开环、
放盒不验证”重写，动作顺序完全一致：

```
首轮引导：左臂桌面视觉伺服抓方块 -> 放入盒子
稳态循环：左臂放入盒子 -> 右臂取出 -> 右臂放回 -> 左臂取出 -> 重复
```

```bash
conda activate hupy3.12
cd /home/zhoulin/hu/square2box/2arm

python dualarm_cycle.py check                 # 只读体检：位姿/串口/相机/YOLO/参数
python dualarm_cycle.py run --cycles 1        # 先跑一轮，人守着急停
python dualarm_cycle.py run --seconds 300     # 5 分钟验收
python dualarm_cycle.py run                   # 无限循环（24/7）
python dualarm_cycle.py home                  # 两臂回 home
python dualarm_cycle.py reset-servo           # 清空学习值，下次重新引导
```

### 它和旧版的区别

| | 旧版 | 新版 |
|---|---|---|
| 盒内抓取 | hover 对一次 pan，之后开环下探 | pan+elbow 两自由度闭环，下探分 4 段逐段跟踪 |
| 视觉目标 | 手标 `align_px` / `box_center_px` | 每次抓成功后自动学习 `hover_px` / `pick_px` + 雅可比 |
| 盲抓 | 视觉失败后退回盲抓 | **没有盲抓**：合爪前必须有新鲜、稳定、在容差内的检测 |
| 下探“没到位” | 抛阶段错误 | 触底/顶住视为正常，停止加深，交给视觉闸门判断 |
| 放盒 | 放完不验证 | 盒内 ROI 复检，偏了就重新夹起再放 |
| 故障恢复 | 固定重跑整轮 | 先重新定位方块（桌面/盒内/夹爪），再从正确状态续跑 |
| 流畅度 | 0.06s settle + 0.8s 到位轮询 + 0.6s 释放停留 | 0.02s settle + 0.25s 到位轮询 + 0.2s 释放停留，下探提速 |

### 学习值 `servo_state.json`

第一次运行时还没有学习值，按标定位姿引导（带 ±4~6° 搜索偏移），
合爪前仍要过视觉闸门；成功一次后自动记录：

- `hover_px`：观察位夹爪正对方块时，方块应出现的像素；
- `pick_px`：抓取位夹爪包住方块时，方块应出现的像素；
- `jacobian`：`d(px,py)/d(pan,elbow)`，在线微探针测量。

之后每次抓取都在 hover 闭环对准、下探逐段闭环跟踪，成功样本用 EMA 持续修正。
文件在 `2arm/servo_state.json`（不纳入 git）。删掉它的效果等于
`python dualarm_cycle.py reset-servo`。

### 关键调参（`config.json` 的 `servo` / `smooth` 段）

| 键 | 默认 | 作用 |
|---|---|---|
| `servo.descent_steps` | 4 | 下探分几段；调大更稳、调小更快 |
| `servo.descent_speed` | 0.5 | 下探速度倍率 |
| `servo.target_tol_px` | 7 | hover 对准容差 |
| `servo.pick_gate_px` | 30 | 合爪前允许的底部像素偏差 |
| `servo.require_visual_before_close` | true | 关掉就等于允许盲抓，**不建议** |
| `servo.pick_retries` | 3 | 单次抓取最多重试几轮 |
| `servo.max_total_deg` | 12 | 视觉修正离标定位姿的最大累计角度 |
| `smooth.arrive_timeout_s` | 0.25 | 到位轮询上限（越小越流畅） |
| `smooth.release_dwell_s` | 0.2 | 松爪后原地停留，落点更稳 |
| `smooth.place_verify` | true | 放盒后盒内复检 |
| `smooth.place_retries` | 1 | 落点偏了重放几次 |

> 注意：`servo.max_total_deg` 限制的是“视觉能把臂拉离标定位姿多远”。
> 方块在盒子里的位移超过这个范围时，servo 会先尝试、失败后由
> `pick_retries` 的搜索偏移补上；再不行就报 `GraspFailed` 进入恢复，
> 不会盲合。

### 24/7 常驻（新版）

```bash
sudo cp dualarm-cycle.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dualarm-cycle
systemctl status dualarm-cycle
journalctl -u dualarm-cycle -f
```

单元里 `Restart=always`、`StartLimitIntervalSec=0`、`KillSignal=SIGINT`，
崩溃/卡死（看门狗 180s 无进展会主动退出）后 5 秒自动拉起；SIGTERM/SIGINT
会先开爪并把夹着的方块放回盒子再下电。

### 离线自测

```bash
python selftest_cycle.py          # 伺服数学 + calib 图真实检测
python selftest_cycle.py servo
python selftest_cycle.py vision
```

---

## 旧版说明（`dualarm_loop.py`，保留作回退）

电机行程标定**不在这个目录里**，一律用官方的 `lerobot-calibrate`
（见第二步 A）。

---

## 第一步：环境

```bash
conda activate hupy3.12
cd /home/zhoulin/hu/square2box/2arm
```

---

## 坐标是怎么来的（先看这段，能省很多时间）

每个关节都归一化到 **−100 … +100**。映射范围来自**官方 lerobot-calibrate
标定出的每条臂的行程**（`config.json` 的 `devices.left_calibration_json` /
`devices.right_calibration_json` 指向那两个文件）：

```
normalized = ( raw - Min ) / ( Max - Min ) * 200 - 100
```

**关键点：raw 超出 [Min, Max] 时会被截断到 ±100。** 一旦截断，
不同的物理高度会变成同一个数值。如果标定范围比手臂实际行程窄，
那么“高位/低位/抓取位”可能全被压成 `+100`，程序以为它们同高，
下探就会一路怼到机械限位。

所以有两条硬规则，程序里都做了拦截：

- `calibrate2arm.py` 记录点位时，只要有关节读数是 ±100 就**拒绝记录**，
  并提示你重摆或先用官方工具重标电机行程。
- `dualarm_loop.py run` 启动前会检查所有点位；有 ±100 的直接拒绝启动，
  并列出是哪几个点、哪个关节。

检查当前状态：

```bash
python calibrate2arm.py --check      # 点位有没有饱和、标定文件和固件是否一致
```

---

## 第二步：全部重新标定（必须先做完）

```bash
python calibrate2arm.py --check       # 0. 只读体检
python calibrate2arm.py --arm left    # 1. 标左臂（夹爪 + 10 个点）
python calibrate2arm.py --arm right   # 2. 标右臂（夹爪 + 10 个点）
python calibrate2arm.py --check       # 3. 复核，应该是 0 个问题
```

### A. 电机行程标定（用官方工具）

电机行程只用官方的 `lerobot-calibrate`，本目录不再提供自写版本。
左臂接在 ACM1（含头部电机，用 leader/teleop 类型），右臂接在 ACM0：

```bash
conda activate hupy3.12
cd /home/zhoulin/hu/square2box/2arm

# 左臂
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=zihao_leader_arm

# 右臂
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=zihao_follower_arm
```

**关键：提示 `Press ENTER to use provided calibration file ... or type 'c'` 时，
取决于你要做什么：**

| 目的 | 按什么 |
|---|---|
| 重新标定行程 | 按 `c` 再回车，然后按提示把每个关节推到底 |
| 把文件里的标定写回电机 | **直接回车**（不按 c） |

第二种正是修“固件退化成 min==max、关节不动”的办法——文件是好的，
只是没写进电机。

标定文件默认落在：
- 左臂 `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/zihao_leader_arm.json`
- 右臂 `~/.cache/huggingface/lerobot/calibration/robots/so_follower/zihao_follower_arm.json`

`config.json` 的 `devices.left_calibration_json` / `devices.right_calibration_json`
就指到这两个文件。换了 `--robot.id` 记得同步改。

标完电机行程，**点位必须全部重标**（归一化坐标系变了）。

### 1. 只读体检（不动电机，安全）

```bash
python calibrate2arm.py --check
```

会检查：官方标定文件是否可读、固件限位是否和文件一致（退化会报出来）、
两个串口、哪些位姿/夹爪还缺、哪些点位被截断。

想看某个关节实时的原始值，用 `--watch`（只读，全程卸力，可以随手推）：

```bash
python calibrate2arm.py --watch --arm left
```

`raw` 停在范围上下限附近推不动 → 机械真的到头了；`raw` 明显超过上下限 →
标定时没扫全行程，重标电机行程。

### 2. 标一条臂（`--arm left` / `--arm right`）

```bash
python calibrate2arm.py --arm left
```

脚本会连接这条臂、**卸力**，然后依次问你：

1. **夹爪**：掰到完全张开，回车；合到刚好夹住方块，回车。
2. **10 个位姿**：把臂摆好，回车。`s` 跳过，`q` 退出。

就这样，和官方 `lerobot-calibrate` 一样——**摆好按回车就行，程序不拦你**。

只有一个可选键：**`t` 加力保持**。手臂卸力时一松手会塌下去，托住摆好姿势后
按 `t` 就能松手再按回车；记完自动卸力。不用也行，托着按回车一样能记。

记完每个点会打一行确认；有问题（读数被截断、离边界太近、两个点重合、
抓取位不竖直）只追加一行 `·` 提示，不会拦你重来。这些留到 `--check` 汇总看。

中途 `Ctrl+C` 或 `q` 退出都会先卸力再断开。

| 点位 | 摆到哪 | 要点 |
|---|---|---|
| `measure` | 完全悬空、四周无物 | 用来测“空夹”读数，必须真的空 |
| `home` | 两臂的休息位 | 避开桌面、盒子、另一条臂 |
| `carry` | 桌面与盒子之间的高处 | 搬运用，别扫到盒子 |
| `table_approach` | 桌上方块正上方、比 hover 高一掌 | 快速段的落点 |
| `table_hover` | 方块正上方，张爪，腕相机能看清方块 | 视觉对准位 |
| `table_pick` | 从 hover **竖直**下到两指能包住方块 | **pan / roll 与 hover 相差 < 2** |
| `box_approach` | 盒口正上方、比 hover 高一掌 | |
| `box_hover` | 盒口正上方，张爪，相机能看清盒内 | 视觉对准位 |
| `box_pick` | 从 hover **竖直**下到能夹住盒内方块 | **pan / roll 与 hover 相差 < 2** |
| `box_drop` | 盒内比 box_pick 略高 1~3，水平位置与 box_pick 相同 | 决定方块落点 |

每个点操作：摆好 → 回车记录。`s` 跳过，`q` 退出。
`table_hover / table_pick / box_hover / box_pick / box_drop` 这几个点会顺手
拍一张带检测框的图（存 `calib/`）。

标 `table_hover` 时会自动记录方块像素位置，作为视觉对准点 `align_px`；
标完 `box_drop`（或 `box_hover`）时，脚本再用检测到的方块位置自动生成
`box_roi`——这是“盒内检测范围”，用来把盒里的方块和桌面上的方块区分开。
盒内暂时没方块时，`box_roi` 以画面中心为锚点。如果盒子不在画面正中，
可以手改 `config.json` 里 `arms.<arm>.gains.box_roi = [x1, y1, x2, y2]`。

标完 `table_pick` / `box_pick` 时，如果 `shoulder_pan` 或 `wrist_roll` 与 hover
相差超过 2，会追加一行提示告诉你“不是竖直下降”。这是 Z 轴垂直拿放的关键，
看到了最好重摆一次（`--only table_pick` 可以只重标这个点）。

`*_approach` 和 `*_hover` 只差不到 3 时也会提示——那说明快速段没有减速余地。

这 10 个点**建议一次标完**。中途退出的话，已记的点会带着“当时的电机标定指纹”
保存；重跑时会问你剩下的点，两套坐标混用的情况 `--check` 会列出来。

### 3. 标右臂

```bash
python calibrate2arm.py --arm right
```

同样的 10 个点。右臂的 `table_*` 也要标（恢复流程会让右臂上桌面）。

### 4. 复核

```bash
python calibrate2arm.py --check          # 这次应该 0 个问题
```

### 只重标某几个点

标错一个点不用重头来：

```bash
python calibrate2arm.py --arm left --only table_hover,table_pick
```

`--only` 只重标列出的位姿，不动夹爪、不重算 `align_px` / `box_roi`。
改完点位后如果有偏差，用 `--arm xxx` 重标一遍最稳。

标定结束时脚本会把**当前电机固件标定的指纹**记进 `config.json`。
下次启动时如果固件标定变了（重标电机、换臂、换机器），程序会明确拒绝
启动并告诉你点位不属于这套标定，而不是带着错坐标去撞限位。

> **这一步之前不要运行 `run`。** 缺点位或者有点位被截断，主程序都会拒绝启动并说明原因。

---

## 第三步：先跑一轮，人守着急停

```bash
python dualarm_loop.py self-test                  # 只读，再确认一次
python dualarm_loop.py preview --arm left --seconds 10   # 看相机+检测
python dualarm_loop.py run --cycles 1             # 只跑一轮
```

看日志里这几行是否正常：

```
[left] 空夹 pos=... load=... 负载可用=True
[left] 闭爪: pos=... (空夹 ...) load=... -> 夹住
[left] 盒内确认到方块
第 1 轮完成，用时 xx.x s
```

箱子放置是否准确，主要看 `box_drop` 标得准不准。若方块落到盒内偏移，
把 `box_drop` 重标（水平位置必须与 `box_pick` 一致）。如果方块被盒底顶住、
夹不出来，把 `box_drop` 抬高 1~3，或把 `grasp.box_drop_extra_lift` 设为 1~2。

> 符号约定：关节值里 **`shoulder_lift` 增大 = 手臂降低**（与 lerobot 归一化一致）。
> 所以所有名字里带 "lift" 的调参项，正数都表示**抬得更高**，程序内部已经帮你取反。

## 第四步：5 分钟 → 24 小时

```bash
# 连续跑 5 分钟（到点自动正常停止）
python dualarm_loop.py run --cycles 0 --seconds 300

# 无限循环（Ctrl+C 会先开爪再下电）
python dualarm_loop.py run --cycles 0
```

### 24/7 常驻（断电重启也能自己起来）

```bash
sudo cp dualarm-loop.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dualarm-loop
systemctl status dualarm-loop
journalctl -u dualarm-loop -f          # 或 tail -f logs/loop.log
```

服务里已配置 `Restart=always`，程序崩溃 5 秒后自动拉起。

---

## 一轮的动作顺序（`dualarm_loop.py` / `dualarm_cycle.py`）

> `four_step_cycle.py` 的顺序见前面的四步说明（它每轮都把方块放回桌面，不依赖 `loop.mode`）。

```
 1. 左臂：桌面抓方块 → 放进盒子
 2. 右臂：从盒子夹出方块
 3. 右臂：放回盒子
 4. 左臂：从盒子夹出方块
    → 方块留在左爪，下一轮直接回到第 1 步的“放进盒子”
```

`loop.mode` 控制收尾：

| 值 | 行为 | 特点 |
|---|---|---|
| `in_box`（默认） | 左臂夹着方块进入下一轮 | 最快、真正往复，无桌面视觉依赖 |
| `table` | 左臂把方块放回桌面，下轮重新抓 | 每轮都有桌面视觉确认，稍慢但更稳 |

---

## 为什么流畅 / 为什么能长时间跑

- **不等、不顿**：每条臂一个后台线程，100Hz 按绝对时间基流式下发目标；
  两条臂可并行，空闲臂会提前“预位”到下一个位姿。
- **释放不驻留**：抬升和松爪同时进行（前 25% 行程内开爪），没有“停住—开爪—再走”的空转。
- **视觉不长时间伺服**：只在 hover 做最多 3 步、每步 ≤3° 的小修正，之后完全按标定位姿走。
- **竖直下探是结构保证的**：下探增量里 `shoulder_pan`、`wrist_roll` 被强制置 0，
  即使视觉把 hover 平移了一点，下降仍然是竖直的。
- **放置精度**：`box_drop` 是单独标定的点（水平与 `box_pick` 相同，略高一点点），
  所以落点和下一轮抓取点是同一个位置，循环不会漂移。
- **24/7 保护**：阶段超时、总线错误重连、相机掉帧重连、电机温度分级（62 警告 / 68 降速 /
  74 卸力冷却）、连续失败硬恢复、每轮写 `logs/cycles_2arm.csv`、定期存快照。

---

## 调参速查（`config.json`）

| 键 | 作用 | 什么时候改 |
|---|---|---|
| `motion.cruise_scale` | 搬运段整体速度 | 想更快/更稳 |
| `motion.precision_scale` | 竖直下探速度 | 下探冲击大就减小 |
| `motion.vmax.<joint>` | 单关节上限 | 某关节抖动/超调时降它 |
| `grasp.align_tol_px` | 视觉对准容差 | 对准总是不够就调大 |
| `grasp.align_step_deg` | 单步对准角度 | 调大更快但更容易过冲 |
| `grasp.retries` | 抓取重试次数 | 方块摆放不稳时调大 |
| `grasp.retry_deepen` | 桌面重试加深量 | 夹不到就调大（每次+0.8） |
| `grasp.box_drop_extra_lift` | 盒内释放**额外抬高**量 | 方块被压住/夹不出（正数=更高） |
| `grasp.table_release_lift` | 桌面释放**额外抬高**量 | 方块砸到桌面（正数=更高） |
| `safety.temp_stop_c` | 温度停机阈值 | 默认 74，慎改 |
| `loop.cycle_pause_s` | 每轮间隔 | 想更快可设 0 |
| `loop.mode` | `in_box` / `table` | 见上表 |
| `arms.<arm>.table_pick_height_offset` | 桌面抓取位再抬高多少（正数=更高） | 抓得太浅、夹不到 |
| `arms.<arm>.box_pick_height_offset` | 盒内抓取位再抬高多少（正数=更高） | 压盒底、夹不到 |

改 `arms.*.poses` 里的数就直接改标定位姿（归一化 −100~100）。

---

## 常见问题

**启动就报“缺少 N 个位姿”** → 还没标完，按报错里的臂跑 `python calibrate2arm.py --arm xxx`。

**`空夹读数偏高`** → `measure` 位被挡（碰到桌面/盒子/线缆），重标该点。

**抓取总是判定“空爪”** → 先看 `load_available` 是否为 True；若为 False 说明该电机
不返回负载，则只能靠夹爪位置判定，需要重新标 `gripper_closed`。若两者都不行，
把 `grasp.pos_margin` 调小一点（例如 0.5）。

**方块放进盒子但位置偏** → `box_drop` 的水平位置和 `box_pick` 不一致，重标 `box_drop`。

**盒子里的方块夹不出来** → `box_drop` / `box_pick` 太深压到盒底，抬高 1~3，或设
`grasp.box_drop_extra_lift`。

**两臂差点撞** → 提高 `carry` 和 `*_approach` 的高度，让两臂路径分开；
盒子区域本身有互锁，同一时刻只允许一条臂进入。

**`table_approach` 和 `table_hover` 只差 0.1** → 录的时候臂没动、空按了两次回车。
重标 `table_approach`，要明确比 `table_hover` 高一个手掌。

**`in_box` 模式下左臂夹着方块挡到右臂** → 把左臂的 `carry` 标到更高、更外侧，
让夹住的方块离盒子正上方足够远；或者改 `loop.mode = "table"`，左臂每轮把方块
放回桌面，两臂之间的空间更宽裕。

**相机 fps 显示 0.0** → 该相机暂时没帧，等到第二帧才有值；持续为 0 就检查
`/dev/video*` 是否被别的进程占用。

**运行中 `BusError`** → 程序会自动重连；反复出现就检查 USB 线和电源。

**启动报“这些位姿的关节读数顶到 ±100，已被标定范围截断”** → 官方标定的行程标窄了，
那个关节摆到范围外时读数被削平成同一个值。先用官方工具重标电机行程：

```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=zihao_leader_arm
python calibrate2arm.py --arm left        # 行程变了，点位必须全部重标
```

**启动报“电机固件里的行程已退化，关节会被锁死动不了”** → 官方标定在“写文件”和
“写固件”之间被打断过：文件是好的，电机里的限位却成了 `min == max`，关节不会动。
不用重标，把文件写回电机即可：

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=zihao_follower_arm
# 提示 "Press ENTER to use provided calibration file ... or type 'c'" 时，
# 直接按【回车】，不要按 c
```

**启动报“点位录制的电机标定与当前硬件不一致”** → 标完点位之后又重标过电机、
换过臂，或从别的机器拷了 `config.json`。旧点位对应的是另一套角度，按提示重标该臂即可。

**`--check` 报“固件限位与标定文件不一致”** → 程序用文件做归一化，但电机按固件
限位执行，不一致会截断运动范围。同样按上面那条“直接回车”写回即可。

**标定点时提示“读数被截断”** → 只有读数真的顶到 ±100（原始值越过标定范围上下限）
才会报。这时高度已经分不开了，用 `--watch` 推一下那个关节：

- `raw` 停在上下限附近推不动 → 机械真的到头了，换个姿态
- `raw` 明显超过上下限 → 标定时没扫全行程，按上面的官方命令重标电机行程，
  然后所有点位重标

只是“离边界不到 5”不会报截断，那是正常的窄余量，照跑。

**`--check` 报“这些点位离标定边界不到 5”** → 没到截断，程序照跑；
但插值和视觉微调会把读数推过边界，那一段就变成“发指令但不动”。看到这条就把
对应关节往行程中间挪一点重标那个点。

**`--check` 报“这些点位没留标定指纹”** → 这些点是加指纹机制之前录的。
程序无法确认它们属于当前电机标定，`run` 会直接拒绝启动。完整重跑一遍该臂即可。

**`four_step_cycle.py` 第 2 步报“盒内看不到方块 / 盒内抓取失败”** → 左臂把方块放偏了
（常见是落在盒子远端边缘，右腕相机看不到、手也够不到）。看 `/tmp/cycle_1_in_box.jpg`
确认落点，把左臂 `box_drop` 重标到右臂 `box_pick` 正上方。

**`four_step_cycle.py` 第 3 步桌面抓空** → 方块被放到盒子边沿、滚进盒里。重标右臂
`table_hover` / `table_pick`（往桌面空处挪，别压盒子边沿、也别贴着腕相机模块），
或加大扫描范围 `--right-pans "8,-8,16,-16,24"`。

**循环里某一轮失败但进程没停** → 这是设计行为：每轮失败会先 `recover_to_start`
（抬离 → home → 右臂桌面点试探抓 → 放回盒子 → 左臂取到桌面点）再继续。用
`[轮次] 第 k 轮失败` 和随后的 `恢复：...` 两行定位是哪一步、恢复有没有成功。

---

## 离线验证（不需要硬件）

```bash
python selftest_offline.py
```

用假臂/假相机跑完整两轮状态机，验证轨迹限速、竖直位姿检查、区域互锁、
空爪重试抛错、目标丢失处理、恢复后区域释放。改完逻辑建议先跑它。
