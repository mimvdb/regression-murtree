from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
OSRT_PATH = (
    SCRIPT_DIR
    / ".."
    / ".."
    / "optimal-sparse-regression-tree-public"
    / "build"
    / "osrt"
)  # SCRIPT_DIR / "osrt"
STREED_PATH = (
    SCRIPT_DIR / ".." / ".." / "streed" / "build" / "STREED"
)  # SCRIPT_DIR / "STREED"
GUIDE_PATH = SCRIPT_DIR / "methods" / "misc" / "guide"
