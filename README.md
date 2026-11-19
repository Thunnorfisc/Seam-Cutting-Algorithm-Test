# Seam carving: greedy, dynamic programming, graph cut

Content-aware image resizing for Assignment 2. Three ways to find the seam to remove, one shared energy function, and a web page to compare them.

## Install

- **Windows:** double-click `install.bat`.
- **Linux / macOS:** `sh install.sh`.
- **Or:** `python -m pip install -r requirements.txt`.

This installs NumPy, OpenCV and Numba, then checks that each one loads and compiles the Numba graph cut once. Only NumPy is required: if OpenCV or Numba cannot load (for example, Windows application control blocks its library), the code falls back and the page leaves out the Numba graph cut.

## Open the page

- **Windows:** double-click `open-page.bat`.
- **Linux / macOS:** `./open-page.sh`.
- **Or:** `python frontend/serve.py`, then open `http://127.0.0.1:8765/`.

It needs Python 3.10 or newer and NumPy (see Install). OpenCV and Numba are optional. Drop a picture in, drag the edges of the carved picture, press Run to compare the methods.

## Where things are

| Path | What |
|---|---|
| `algorithms/greedy.py`, `greedy_resize.py` | Greedy seams |
| `algorithms/dp.py`, `dp_numpy.py`, `dp_resize.py` | Dynamic-programming seams |
| `algorithms/graphcut.py`, `graphcut_resize.py` | Graph-cut seams (Edmonds-Karp and Dinic) |
| `algorithms/graphcut_numba.py` | The same Dinic compiled with Numba |
| `install.bat`, `install.sh`, `tools/check_setup.py` | Install the packages and check they load |
| `temporary/` | Shared energy map, seam removal and image files |
| `benchmarks/` | Command-line comparisons |
| `frontend/` | The web page and its server |
| `tests/` | Unit tests |
| `docs/` | The write-up for each method, and the notes for teammates |

## Run the tests

```text
python -m unittest discover -s tests
```

Every push and pull request runs them on Linux and Windows (`.github/workflows/ci.yml`). When they pass on `main`, the run page has the packaged zip to download.
