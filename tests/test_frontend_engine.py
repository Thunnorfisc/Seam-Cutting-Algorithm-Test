"""Tests for the web page's back end: the engine functions and the HTTP API."""

from __future__ import annotations

import base64
import json
import random
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np

from algorithms.dp_resize import resize_dp
from algorithms.graphcut_resize import resize_graphcut
from algorithms.greedy_resize import resize_greedy
from frontend import engine
from frontend.serve import make_server


def noise_bgr(seed: int, height: int, width: int) -> np.ndarray:
    rng = random.Random(seed)
    values = [rng.randrange(256) for _ in range(height * width * 3)]
    return np.array(values, dtype=np.uint8).reshape(height, width, 3)


class EngineTests(unittest.TestCase):
    def test_pixel_round_trip_swaps_red_and_blue(self) -> None:
        rgba = bytes([10, 20, 30, 255, 40, 50, 60, 0])
        bgr = engine.rgba_to_bgr(rgba, 2, 1)
        self.assertEqual(bgr.tolist(), [[[30, 20, 10], [60, 50, 40]]])
        self.assertEqual(engine.bgr_to_rgba(bgr), bytes([10, 20, 30, 255, 40, 50, 60, 255]))
        with self.assertRaises(ValueError):
            engine.rgba_to_bgr(rgba, 3, 1)

    def test_trace_reproduces_each_resize_function(self) -> None:
        image = noise_bgr(1, 9, 10)
        expected = {
            "greedy": resize_greedy(image, 7, 6),
            "dp": resize_dp(image, 7, 6),
            "dp_numpy": resize_dp(image, 7, 6),
            "graphcut_dinic": resize_graphcut(image, 7, 6),
        }
        for name, want in expected.items():
            result = engine.carve_trace(image, 7, 6, name)
            self.assertTrue(np.array_equal(result["image"], want), name)
            self.assertEqual((result["vertical_seams"], result["horizontal_seams"]), (3, 3))

    def test_order_map_accounts_for_every_removed_pixel(self) -> None:
        image = noise_bgr(2, 8, 11)
        before = image.copy()
        result = engine.carve_trace(image, 8, 5, "dp")
        order = result["order"]
        self.assertEqual(order.shape, (8, 11))
        self.assertTrue(np.array_equal(image, before))
        # A vertical seam takes one pixel from each of the 8 rows.
        for k in range(1, 4):
            self.assertEqual(int((order == k).sum()), 8)
            self.assertTrue(((order == k).sum(axis=1) == 1).all())
        # A horizontal seam takes one pixel from each of the 8 columns left.
        for k in range(4, 7):
            self.assertEqual(int((order == k).sum()), 8)
        # The survivors, read in order, are the carved image.
        kept = image[order == 0].reshape(5, 8, 3)
        self.assertEqual(int((order == 0).sum()), 5 * 8)
        self.assertTrue(np.array_equal(np.sort(kept.reshape(-1, 3), axis=0),
                                       np.sort(result["image"].reshape(-1, 3), axis=0)))

    def test_first_seams_agree_on_the_minimum(self) -> None:
        image = noise_bgr(3, 10, 12)
        rows = {name: engine.first_seams(image, name, repeats=1)
                for name in ("greedy", "dp", "dp_numpy", "graphcut_dinic", "graphcut_edmonds_karp")}
        for index, orientation in enumerate(("vertical", "horizontal")):
            dp = rows["dp"][index]
            self.assertEqual(dp["orientation"], orientation)
            self.assertEqual(rows["dp_numpy"][index]["seam"], dp["seam"])
            self.assertAlmostEqual(rows["graphcut_dinic"][index]["cost"], dp["cost"], places=6)
            self.assertAlmostEqual(rows["graphcut_edmonds_karp"][index]["cost"], dp["cost"], places=6)
            self.assertLessEqual(dp["cost"], rows["greedy"][index]["cost"] + 1e-9)

    def test_progress_can_cancel(self) -> None:
        seen = []

        def progress(done, total):
            seen.append((done, total))
            if done == 2:
                raise engine.Cancelled()

        with self.assertRaises(engine.Cancelled):
            engine.carve_trace(noise_bgr(4, 6, 8), 3, 6, "dp", progress)
        self.assertEqual(seen, [(1, 5), (2, 5)])

    def test_size_limit_and_bad_input(self) -> None:
        big = np.zeros((200, 200, 3), dtype=np.uint8)
        with self.assertRaises(engine.TooLarge):
            engine.check_size("graphcut_dinic", big)
        engine.check_size("graphcut_dinic", big, force=True)
        engine.check_size("dp", big)
        with self.assertRaises(ValueError):
            engine.check_size("nope", big)
        with self.assertRaises(ValueError):
            engine.carve_trace(noise_bgr(5, 4, 4), 5, 4, "dp")

    def test_flat_image_energy_preview(self) -> None:
        data, peak = engine.energy_preview(np.full((3, 4, 3), 80, dtype=np.uint8))
        self.assertEqual(peak, 0.0)
        self.assertEqual(data, bytes(12))


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = make_server(0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def post(self, path: str, body: dict):
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.loads(error.read())

    def upload(self, image: np.ndarray) -> str:
        height, width = image.shape[:2]
        status, reply = self.post("/api/image", {
            "width": width, "height": height,
            "rgba": base64.b64encode(engine.bgr_to_rgba(image)).decode(),
        })
        self.assertEqual(status, 200)
        self.assertEqual(len(base64.b64decode(reply["energy"])), width * height)
        return reply["id"]

    def test_page_is_served(self) -> None:
        with urllib.request.urlopen(self.base + "/", timeout=10) as response:
            self.assertIn(b"Seam carving", response.read())

    def test_full_round_trip(self) -> None:
        image = noise_bgr(6, 8, 10)
        image_id = self.upload(image)
        status, methods = self.post("/api/methods", {})
        self.assertEqual(status, 200)
        self.assertIn("dp", [m["name"] for m in methods["methods"]])

        status, first = self.post("/api/first", {"id": image_id, "method": "dp"})
        self.assertEqual(status, 200)
        self.assertEqual([row["orientation"] for row in first["rows"]], ["vertical", "horizontal"])

        status, carved = self.post("/api/carve", {"id": image_id, "method": "dp",
                                                  "target_width": 7, "target_height": 6})
        self.assertEqual(status, 200)
        self.assertEqual((carved["width"], carved["height"]), (7, 6))
        got = engine.rgba_to_bgr(base64.b64decode(carved["rgba"]), 7, 6)
        self.assertTrue(np.array_equal(got, resize_dp(image, 7, 6)))
        order = np.frombuffer(base64.b64decode(carved["order"]), dtype="<u2")
        self.assertEqual(order.size, 80)
        self.assertEqual(int((order == 0).sum()), 42)

    def test_errors_are_reported_as_json(self) -> None:
        image_id = self.upload(noise_bgr(7, 5, 5))
        self.assertEqual(self.post("/api/first", {"id": "missing", "method": "dp"})[0], 400)
        self.assertEqual(self.post("/api/first", {"id": image_id, "method": "nope"})[0], 400)
        self.assertEqual(self.post("/api/carve", {"id": image_id, "method": "dp",
                                                  "target_width": 9, "target_height": 5})[0], 400)
        self.assertEqual(self.post("/api/carve", {"id": image_id, "method": "dp",
                                                  "target_width": "3", "target_height": 5})[0], 400)
        self.assertEqual(self.post("/api/image", {"width": 2, "height": 2, "rgba": "AAAA"})[0], 400)
        self.assertEqual(self.post("/api/nothing", {})[0], 404)

    def test_graph_cut_is_refused_on_a_large_image(self) -> None:
        image_id = self.upload(np.zeros((150, 150, 3), dtype=np.uint8))
        status, reply = self.post("/api/first", {"id": image_id, "method": "graphcut_dinic"})
        self.assertEqual(status, 400)
        self.assertTrue(reply["too_large"])


if __name__ == "__main__":
    unittest.main()
