"""
环境自检脚本

跑这个脚本可以一次性检查:
  - SC2 安装目录是否正确
  - SC2 可执行文件是否存在
  - 地图是否在位
  - 关键依赖能否 import
  - burnysc2 能否解析 SC2PATH

用法:
    .venv/Scripts/python.exe scripts/check_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 把项目根加进 sys.path,这样直接跑也能 import bot.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.config import SC2_INSTALL_DIR, AVAILABLE_MAPS  # noqa: E402

OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(label: str, cond: bool, detail: str = "") -> bool:
    tag = OK if cond else FAIL
    line = f"{tag} {label}"
    if detail:
        line += f"  -- {detail}"
    print(line)
    return cond


def main() -> int:
    print("=" * 60)
    print("SC2 AI Bot - 环境自检")
    print("=" * 60)

    all_ok = True

    # 1. Python 版本
    py = sys.version_info
    check(
        f"Python 版本 {py.major}.{py.minor}.{py.micro}",
        py.major == 3 and py.minor >= 10,
        f"建议 >=3.10, 当前 {py.major}.{py.minor}",
    )

    # 2. SC2 安装目录
    sc2_dir_ok = SC2_INSTALL_DIR.is_dir()
    all_ok &= check(
        f"SC2 安装目录存在",
        sc2_dir_ok,
        str(SC2_INSTALL_DIR),
    )

    # 3. SC2 可执行文件
    if sc2_dir_ok:
        exe = SC2_INSTALL_DIR / "Versions"
        versions = sorted(exe.glob("Base*")) if exe.exists() else []
        has_versions = len(versions) > 0
        all_ok &= check(
            f"SC2 Versions 目录",
            has_versions,
            f"找到 {len(versions)} 个 Base* 版本: "
            + ", ".join(v.name for v in versions),
        )

        # 找 SC2_x64.exe 之类的
        sc2_exe_candidates = list((SC2_INSTALL_DIR / "Versions").rglob("SC2*.exe")) if exe.exists() else []
        all_ok &= check(
            f"SC2 可执行文件",
            len(sc2_exe_candidates) > 0,
            f"找到 {len(sc2_exe_candidates)} 个: "
            + ", ".join(p.name for p in sc2_exe_candidates[:3]),
        )

    # 4. Maps 目录
    maps_dir = SC2_INSTALL_DIR / "Maps"
    maps_ok = maps_dir.is_dir()
    all_ok &= check(f"Maps 目录", maps_ok, str(maps_dir))

    if maps_ok:
        found_maps = {p.stem for p in maps_dir.rglob("*.SC2Map")}
        for mapname in AVAILABLE_MAPS:
            all_ok &= check(
                f"  地图: {mapname}",
                mapname in found_maps,
            )

    # 5. 关键 Python 依赖
    print()
    print("--- Python 依赖 ---")

    for mod_name in ["sc2", "numpy", "loguru", "aiohttp", "google.protobuf"]:
        try:
            __import__(mod_name)
            check(f"import {mod_name}", True)
        except ImportError as e:
            all_ok &= check(f"import {mod_name}", False, str(e))

    # 6. SC2PATH 环境变量
    import os

    sc2path_env = os.environ.get("SC2PATH", "")
    check(
        "SC2PATH 环境变量",
        sc2path_env == str(SC2_INSTALL_DIR),
        f"= {sc2path_env}",
    )

    # 7. burnysc2 是否能自己定位
    try:
        from sc2 import paths  # type: ignore

        resolved = paths.Paths.BASE
        all_ok &= check(
            "burnysc2 能定位 SC2",
            resolved == SC2_INSTALL_DIR,
            f"resolved = {resolved}",
        )
    except Exception as e:
        all_ok &= check("burnysc2 能定位 SC2", False, repr(e))

    # 总结
    print("=" * 60)
    if all_ok:
        print(f"{OK} 全部检查通过,可以尝试启动 Bot 了。")
        print()
        print("下一步:  .venv/Scripts/python.exe run.py")
        return 0
    else:
        print(f"{FAIL} 有失败项,请先修复再继续。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
