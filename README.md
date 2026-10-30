# square2box

SO-101 / LeRobot 左臂视觉抓取与循环放置。

## 文件

- `square2box.py` — 单次抓取/放置 + 手动标定（HSV + 视觉伺服）
- `square2box_loop.py` — 24/7 循环：桌面抓取 → 放入盆内 → 盆内抓出 → 放回桌面；YOLOE + HSV、平滑运动、相机/总线看门狗、温度保护
- `calibrate_gripper.py` — 单独重新标定夹爪开合
- `README_LOOP.md` — 详细使用、标定、参数和排障
- `square2box-loop.service` — systemd 常驻配置
- `config.json` — 位姿/增益/夹爪标定参数

## 环境

```bash
conda activate hupy3.12
pip install -r requirements.txt
```

本项目使用本地 Lerobot 的 `XLerobot`（左臂 bus），请按你的 LeRobot 安装方式装好 `lerobot`。YOLOE 资产不放进仓库：

```bash
# 放在项目根目录（或用软链接）
yoloe-11l-seg.pt
mobileclip_blt.ts
```

## 快速开始

```bash
python square2box_loop.py self-test
python calibrate_gripper.py --read
python square2box_loop.py run --cycles 1
python square2box_loop.py run --cycles 0
```

详细标定、参数和常见问题见 [README_LOOP.md](README_LOOP.md)。

## 设备

脚本优先使用 `/dev/left_arm`、`/dev/right_arm`、`/dev/left_wrist_cam`；符号链接缺失时自动回退到：

- 左臂（8 电机含头）：`/dev/ttyACM1`
- 右臂（6 电机）：`/dev/ttyACM0`
- 左腕相机：`/dev/video2`
