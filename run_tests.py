"""Minimal test runner: no network here to pip install pytest, so this
discovers test_*.py files under tests/, runs every test_* function, and
reports pass/fail with tracebacks. Swap for `pytest` in CI where available."""
import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
TESTS_DIR = ROOT / "tests"


def discover():
    return sorted(TESTS_DIR.glob("test_*.py"))


def run():
    total = 0
    failed = 0
    for path in discover():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn_names = [n for n in dir(mod) if n.startswith("test_")]
        for name in fn_names:
            total += 1
            fn = getattr(mod, name)
            try:
                fn()
                print(f"PASS  {path.name}::{name}")
            except Exception:
                failed += 1
                print(f"FAIL  {path.name}::{name}")
                traceback.print_exc()
    print(f"\n{total - failed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
