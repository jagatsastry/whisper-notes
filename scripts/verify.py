#!/usr/bin/env python3
"""
Verify Quill is running correctly as a menu bar dictation app.

Usage:
    uv run python scripts/verify.py          # check running app
    uv run python scripts/verify.py --all    # include tests + lint
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
results: list[tuple[str, bool]] = []


def check(name, passed, detail=""):
    results.append((name, passed))
    mark = PASS if passed else FAIL
    msg = f"  {mark} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return passed


def run(cmd, timeout=10, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)


def osascript(script):
    return run(["osascript", "-e", script])


def check_process():
    print("\n=== Process ===")
    ps = run(["pgrep", "-f", "from quill.app import main"])
    pids = [p for p in ps.stdout.strip().split("\n") if p]
    check("App process running", len(pids) > 0, f"PIDs: {pids}" if pids else "not found")

    result = osascript(
        'tell application "System Events" to return name of every process '
        "whose visible is true"
    )
    in_dock = "Python" in result.stdout
    check("Python NOT in Dock", not in_dock)


def check_menu_bar():
    print("\n=== Menu Bar ===")
    result = osascript(
        'tell application "System Events"\n'
        '  tell process "Python"\n'
        '    return title of every menu bar item of menu bar 1\n'
        "  end tell\n"
        "end tell"
    )
    has_menu = "Quill" in result.stdout
    check("Menu bar shows Quill", has_menu, result.stdout.strip() or result.stderr.strip())

    if not has_menu:
        print(f"  {WARN} Skipping menu content checks (no menu bar item)")
        return

    result = osascript(
        'tell application "System Events"\n'
        '  tell process "Python"\n'
        "    return name of every menu item of menu 1 "
        "of menu bar item 1 of menu bar 1\n"
        "  end tell\n"
        "end tell"
    )
    items = result.stdout.strip()
    check("Menu has 'Enable Dictation'", "Dictation" in items, items)
    check("Menu has 'Quit'", "Quit" in items)
    check("Menu hides 'Start Recording' (flag off)", "Start Recording" not in items)
    check("Menu hides 'Live Transcribe' (flag off)", "Live Transcribe" not in items)


def check_screenshot():
    print("\n=== Screenshot ===")
    screenshot = "/tmp/quill-verify-menubar.png"
    run(["screencapture", "-x", screenshot, "-R", "0,0,2560,40"])
    exists = Path(screenshot).exists()
    check("Menu bar screenshot saved", exists, screenshot)


def check_config():
    print("\n=== Config ===")
    result = run(
        [
            sys.executable,
            "-c",
            "from quill.config import Config; c = Config(); "
            "assert c.enable_transcription is False, 'transcription should be off'; "
            "assert c.enable_summarization is False, 'summarization should be off'; "
            "assert c.dictation_model == 'large-v3', f'model={c.dictation_model}'; "
            "assert c.dictation_hotkey == 'alt_r'; "
            "print('ok')",
        ],
        timeout=30,
    )
    check(
        "Default config correct",
        result.stdout.strip() == "ok",
        result.stderr.strip() if result.returncode else "dictation-only, large-v3",
    )

    # Check model cache
    from quill.config import Config

    cfg = Config()
    fw_cache = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--Systran--faster-whisper-{cfg.dictation_model}"
    )
    cached = fw_cache.exists()
    if cached:
        check("Dictation model cached", True, cfg.dictation_model)
    else:
        print(
            f"  {WARN} Model '{cfg.dictation_model}' not cached — "
            "will download on first dictation use"
        )


def check_tests():
    print("\n=== Tests ===")
    result = run([sys.executable, "-m", "pytest", "tests/", "-x", "-q"], timeout=120)
    last = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else ""
    check("Test suite passes", "passed" in last and "failed" not in last, last)


def check_lint():
    print("\n=== Lint ===")
    result = run([sys.executable, "-m", "ruff", "check", "quill/", "tests/"])
    check("Ruff lint clean", result.returncode == 0, result.stdout.strip() or "clean")


def main():
    parser = argparse.ArgumentParser(description="Verify Quill app")
    parser.add_argument("--all", action="store_true", help="Include tests + lint")
    args = parser.parse_args()

    print("Quill Verification")

    check_process()
    check_menu_bar()
    check_screenshot()
    check_config()

    if args.all:
        check_tests()
        check_lint()

    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\n{'=' * 40}")
    print(f"  {passed}/{total} checks passed")
    if passed < total:
        print(f"  {total - passed} FAILED:")
        for name, p in results:
            if not p:
                print(f"    - {name}")
    print(f"{'=' * 40}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
