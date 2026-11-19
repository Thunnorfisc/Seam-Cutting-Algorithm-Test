"""Local web server for the seam carving comparison page.

    python frontend/serve.py            (Windows: py frontend\\serve.py)

or double-click ``open-page.bat`` (Windows) / run ``./open-page.sh`` in the
repository root. The page opens in the browser at http://127.0.0.1:8765/. Standard library only: the browser decodes
the picture and sends raw pixels, so OpenCV is not needed and PNG or JPEG
files work on a machine where only BMP loads in Python.

The server listens on 127.0.0.1 only and keeps the last few uploaded images
in memory. Nothing is written to disk.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import uuid
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from algorithms.greedy_resize import REMOVAL_ORDER
from frontend import engine

PAGE = Path(__file__).resolve().with_name("index.html")
MAX_BODY = 96 * 1024 * 1024
MAX_SIDE = 4096
KEPT_IMAGES = 4

_lock = threading.Lock()
_images: "OrderedDict[str, np.ndarray]" = OrderedDict()
_jobs: dict[str, dict] = {}


class BadRequest(Exception):
    pass


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _image(image_id) -> np.ndarray:
    with _lock:
        image = _images.get(image_id)
    if image is None:
        raise BadRequest("image not found; load it again")
    return image


def _int(body: dict, key: str) -> int:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRequest(f"{key} must be an integer")
    return value


def api_methods(body: dict) -> dict:
    return {
        "methods": engine.method_table(),
        "energy_formula": engine.ENERGY_FORMULA,
        "removal_order": REMOVAL_ORDER,
    }


def api_image(body: dict) -> dict:
    width, height = _int(body, "width"), _int(body, "height")
    if not (1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE):
        raise BadRequest(f"image sides must be between 1 and {MAX_SIDE}")
    try:
        data = base64.b64decode(body.get("rgba", ""), validate=True)
        image = engine.rgba_to_bgr(data, width, height)
    except (ValueError, TypeError) as error:
        raise BadRequest(f"bad pixel data: {error}") from error
    image_id = uuid.uuid4().hex
    with _lock:
        _images[image_id] = image
        while len(_images) > KEPT_IMAGES:
            _images.popitem(last=False)
    energy, peak = engine.energy_preview(image)
    return {"id": image_id, "width": width, "height": height,
            "energy": _b64(energy), "energy_max": peak}


def api_first(body: dict) -> dict:
    image = _image(body.get("id"))
    name = body.get("method")
    engine.check_size(name, image, bool(body.get("force")))
    slow = engine.METHODS[name]["max_pixels"] is not None
    return {"rows": engine.first_seams(image, name, repeats=1 if slow else 3)}


def api_carve(body: dict) -> dict:
    image = _image(body.get("id"))
    name = body.get("method")
    engine.check_size(name, image, bool(body.get("force")))
    job_id = str(body.get("job") or uuid.uuid4().hex)
    job = {"done": 0, "total": 0, "cancel": False}
    with _lock:
        _jobs[job_id] = job

    def progress(done: int, total: int) -> None:
        job["done"], job["total"] = done, total
        if job["cancel"]:
            raise engine.Cancelled()

    try:
        result = engine.carve_trace(image, _int(body, "target_width"),
                                    _int(body, "target_height"), name, progress)
    except engine.Cancelled:
        return {"cancelled": True, "done": job["done"]}
    finally:
        with _lock:
            _jobs.pop(job_id, None)

    carved = result["image"]
    return {
        "method": name,
        "width": int(carved.shape[1]),
        "height": int(carved.shape[0]),
        "rgba": _b64(engine.bgr_to_rgba(carved)),
        # One little-endian uint16 per ORIGINAL pixel: 0 kept, k = removed by seam k.
        "order": _b64(result["order"].astype("<u2").tobytes()),
        "vertical_seams": result["vertical_seams"],
        "horizontal_seams": result["horizontal_seams"],
        "seams_removed": result["seams_removed"],
        "total_seconds": result["total_seconds"],
        "find_seconds": result["find_seconds"],
        "energy_seconds": result["energy_seconds"],
    }


def api_progress(body: dict) -> dict:
    with _lock:
        job = _jobs.get(str(body.get("job")))
    if job is None:
        return {"known": False}
    return {"known": True, "done": job["done"], "total": job["total"]}


def api_cancel(body: dict) -> dict:
    with _lock:
        job = _jobs.get(str(body.get("job")))
    if job is not None:
        job["cancel"] = True
    return {"known": job is not None}


ROUTES = {
    "/api/methods": api_methods,
    "/api/image": api_image,
    "/api/first": api_first,
    "/api/carve": api_carve,
    "/api/progress": api_progress,
    "/api/cancel": api_cancel,
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # keep the terminal quiet
        pass

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        handler = ROUTES.get(self.path)
        if handler is None:
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            self.close_connection = True
            self._json(413, {"error": "request too large"})
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise BadRequest("request body must be a JSON object")
            self._json(200, handler(body))
        except engine.TooLarge as error:
            self._json(400, {"error": str(error), "too_large": True})
        except (BadRequest, ValueError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:  # report it to the page instead of dropping the connection
            self._json(500, {"error": f"{type(error).__name__}: {error}"})

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def make_server(port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Seam carving comparison page")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the page in the browser")
    args = parser.parse_args(argv)
    try:
        server = make_server(args.port)
    except OSError:
        # Most likely this server is already running: just show its page.
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Port {args.port} is in use. Opening {url} in case the page is already running.")
        if not args.no_browser:
            webbrowser.open(url)
        return
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Seam carving page: {url}  (Ctrl+C or close this window to stop)")
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
