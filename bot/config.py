"""
配置 SC2 路径（国服客户端在非标准位置）
所有脚本入口都先 `from bot.config import *` 来确保路径正确。
"""
from __future__ import annotations

import os
from pathlib import Path

# 项目根目录: sc2-ai-bot/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SC2 安装目录: 跟项目同级的 StarCraft II
SC2_INSTALL_DIR = PROJECT_ROOT.parent / "StarCraft II"

# 关键: burnysc2 通过 SC2PATH 环境变量定位国服客户端
os.environ["SC2PATH"] = str(SC2_INSTALL_DIR)

# 默认地图（Ladder 2019 Season 3，已下载到 Maps 根目录）
DEFAULT_MAP = "AcropolisLE"

AVAILABLE_MAPS = [
    "AcropolisLE",
    "DiscoBloodbathLE",
    "EphemeronLE",
    "ThunderbirdLE",
    "TritonLE",
    "WintersGateLE",
    "WorldofSleepersLE",
]
