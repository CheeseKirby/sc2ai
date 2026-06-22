"""
Win32 window hiding utilities for SC2 / Battle.net.

Hides target processes' top-level windows by calling ShowWindow(hwnd, SW_HIDE),
so they disappear from the taskbar and Alt+Tab list while the process keeps running.

Used by:
  - scripts/hide_sc2.py: one-shot hide
  - scripts/unhide_sc2.py: restore
  - scripts/window_guard.py: continuous watchdog
  - run.py: hide after bot launches SC2
"""
from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes

logger = logging.getLogger(__name__)

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

SW_HIDE = 0
SW_SHOW = 5
SW_RESTORE = 9

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

TARGET_PROCESSES: frozenset[str] = frozenset(
    {
        "sc2_x64.exe",
        "sc2.exe",
        "starcraft ii.exe",
        "battle.net.exe",
    }
)

EnumWindowsProc = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
)


def get_process_name(pid: int) -> str:
    """Return the basename of the executable that owns ``pid``, or "" on failure."""
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        ok = kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        if not ok:
            return ""
        return buf.value.rsplit("\\", 1)[-1]
    finally:
        kernel32.CloseHandle(handle)


def get_window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def hide_target_windows(
    targets: frozenset[str] = TARGET_PROCESSES,
) -> int:
    """Hide every visible top-level window owned by any of ``targets``.

    Returns the number of windows hidden.
    """
    hidden = 0

    def callback(hwnd: int, _lparam: int) -> bool:
        nonlocal hidden
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        name = get_process_name(pid.value).lower()
        if name in targets:
            title = get_window_title(hwnd)
            user32.ShowWindow(hwnd, SW_HIDE)
            hidden += 1
            logger.info(
                "Hidden hwnd=%d pid=%d name=%s title=%r",
                hwnd,
                pid.value,
                name,
                title,
            )
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return hidden


def show_target_windows(
    targets: frozenset[str] = TARGET_PROCESSES,
) -> int:
    """Restore (show + un-minimize) every top-level window owned by ``targets``.

    Skips windows with empty titles to avoid resurrecting helper/tool windows.
    Returns the number of windows restored.
    """
    shown = 0

    def callback(hwnd: int, _lparam: int) -> bool:
        nonlocal shown
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        name = get_process_name(pid.value).lower()
        if name not in targets:
            return True
        title = get_window_title(hwnd)
        if not title:
            return True
        user32.ShowWindow(hwnd, SW_SHOW)
        user32.ShowWindow(hwnd, SW_RESTORE)
        shown += 1
        logger.info(
            "Restored hwnd=%d pid=%d name=%s title=%r",
            hwnd,
            pid.value,
            name,
            title,
        )
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return shown
