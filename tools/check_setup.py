"""Check that this project's packages import, and compile the Numba graph cut.

install.bat / install.sh run this after pip. NumPy is required. OpenCV and
Numba are optional: without OpenCV the code uses its NumPy fallback, and
without Numba the web page leaves out the Numba graph cut. Exit code 1 only
when NumPy is missing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check(label: str, module_name: str, if_missing: str):
    """Import ``module_name`` and report its version, or why it is not available. Returns the module or ``None``."""
    try:
        module = __import__(module_name)
    except Exception as error:  # ImportError, or OSError when Windows blocks the library
        print(f"  {label:7s} not available ({type(error).__name__}: {error})")
        print(f"          {if_missing}")
        return None
    print(f"  {label:7s} OK, version {getattr(module, '__version__', 'unknown')}")
    return module


def main() -> int:
    """Check every package, compile the Numba code when Numba loads, and return the exit code."""
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    numpy = check("NumPy", "numpy", "Required. Fix the error above and run the installer again.")
    check("OpenCV", "cv2", "Optional: energy and BMP files use the NumPy fallback, and the web page works.")
    numba = check("Numba", "numba", "Optional: the web page leaves out the Numba graph cut; every other method works.")
    if numpy is None:
        return 1

    if numba is not None:
        print("\nCompiling the Numba graph cut (a few seconds the first time) ...", flush=True)
        try:
            from algorithms import graphcut_numba
            start = time.perf_counter()
            graphcut_numba.warm_up()
            print(f"  done in {time.perf_counter() - start:.1f} s. Later starts load it from the cache.")
        except Exception as error:
            print(f"  could not compile ({type(error).__name__}: {error})")
            print("  The web page will leave out the Numba graph cut.")

    print("\nReady. Start the page with open-page.bat (Windows) or ./open-page.sh (Linux / macOS).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
