#!/usr/bin/env bash
# 用 hupy3.12 环境启动标定后巡演；透传所有参数。
set -euo pipefail
cd "$(dirname "$0")"

PY=/home/zhoulin/miniconda3/envs/hupy3.12/bin/python
if [[ ! -x "$PY" ]]; then
  PY=python3
fi
exec "$PY" run_calibrated_flow.py "$@"
