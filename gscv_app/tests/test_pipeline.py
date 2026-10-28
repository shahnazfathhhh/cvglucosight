"""Tes tanpa internet dan tanpa torch:  python tests/test_pipeline.py"""
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gscv import gemini  # noqa: E402
from gscv.config import TABLE_PATH  # noqa: E402
from gscv.pipeline import analyze, format_result, from_rows, to_cv_record  # noqa: E402
from gscv.table import FoodTable  # noqa: E402

OK = FAIL = 0


def check(name, cond, detail=""):
    global OK, FAIL
    OK, FAIL = (OK + 1, FAIL) if cond else (OK, FAIL + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {'' if cond else detail}")


T = FoodTable(TABLE_PATH)
tmp = Path(tempfile.mkdtemp())
img = tmp / "meal.jpg"
Image.fromarray((np.random.default_rng(0).random((300, 400, 3)) * 255).astype("uint8")).save(img)


class FakeLocal:
    def __init__(self, cls, conf):
        self.cls, self.conf = cls, conf

    def predict_topk(self, p, k=1):
        return [(self.cls, self.conf)]


print("[1] Tabel & perkalian")
check("10 kelas", len(T.classes) == 10)
check("nasi_goreng 200 g -> 54.0 g karbo", T.macros("nasi_goreng", 200)["carbs_g"] == 54.0)
check("linear 3x", abs(T.macros("soto", 300)["carbs_g"] - 3 * T.macros("soto", 100)["carbs_g"]) < 0.2)

print("[2] Jalur Gemini (VLM palsu)")
vlm = lambda p, c: {"is_food": True, "items": [  # noqa: E731
    {"name": "Nasi goreng", "table_class": "nasi_goreng", "grams": 300, "grams_low": 250, "grams_high": 350, "confidence": 0.9},
    {"name": "Mie goreng", "table_class": "other", "grams": 200, "confidence": 0.8},
    {"name": "Rusak", "table_class": "soto", "grams": "banyak", "confidence": 0.9},
    {"name": "Terlalu besar", "table_class": "bakso", "grams": 99999, "confidence": 0.9},
]}
r = analyze(img, T, vlm=vlm)
check("engine gemini", r["engine"] == "gemini")
check("gram tidak valid dilewati", len(r["items"]) == 3)
check("karbo nasi goreng 300 g = 81.0", r["items"][0]["macros"]["carbs_g"] == 81.0)
check("item di luar tabel -> needs_manual_input", r["needs_manual_input"] and not r["total_complete"])
check("gram ekstrem dipotong ke 1000", r["items"][2]["grams"] == 1000.0)
check("total hanya dari item di tabel",
      r["totals"]["carbs_g"] == round(r["items"][0]["macros"]["carbs_g"] + r["items"][2]["macros"]["carbs_g"], 1))
rec = to_cv_record(r)
check("carbs_source vlm_grams_x_table", rec["carbs_source"] == "vlm_grams_x_table")
check("format_result jalan", "Total" in format_result(r))

r = analyze(img, T, vlm=lambda p, c: {"is_food": False, "items": []})
check("bukan makanan -> manual", r["needs_manual_input"] and r["items"] == [])

print("[3] Fallback")
def boom(p, c):
    raise gemini.GeminiError("HTTP 503")
r = analyze(img, T, vlm=boom, local=FakeLocal("soto", 0.93))
check("Gemini gagal -> lokal", r["engine"] == "local" and r["items"][0]["grams"] == 250.0)
check("lokal yakin: tidak perlu manual", not r["needs_manual_input"])
check("lokal memberi catatan gram = porsi standar", any("porsi standar" in w for w in r["warnings"]))
r = analyze(img, T, vlm=None, local=FakeLocal("pempek", 0.40))
check("lokal conf 0.40 -> manual", r["needs_manual_input"])
r = analyze(img, T, vlm=None, local=None)
check("tanpa mesin -> manual, tidak crash", r["engine"] == "none" and r["needs_manual_input"])
r = analyze(img, T, vlm=lambda p, c: ["bukan", "dict"], local=FakeLocal("soto", 0.9))
check("format VLM salah -> lokal", r["engine"] == "local")

print("[4] Koreksi manual")
r = from_rows([["Mie goreng", "nasi goreng", 300], ["", "", None], ["Soto", "soto", "abc"], ["Gudeg", "gudeg", 250]], T)
check("spasi pada nama kelas dinormalkan", r["items"][0]["food_class"] == "nasi_goreng")
check("baris kosong dilewati & gram salah dilaporkan", len(r["items"]) == 2 and len(r["warnings"]) == 1)
check("gudeg 250 g = 42.5 g karbo", r["items"][1]["macros"]["carbs_g"] == 42.5)
check("manual lengkap -> tidak perlu cek", not r["needs_manual_input"])
check("carbs_source user_grams_x_table", to_cv_record(r)["carbs_source"] == "user_grams_x_table")
r = from_rows([["Pizza", "pizza", 200]], T)
check("kelas tak dikenal ditandai", r["needs_manual_input"] and r["items"][0]["macros"] is None)

print("[5] Klien Gemini (HTTP dipalsukan)")
class Resp:
    def __init__(self, status, payload):
        self.status_code, self._p, self.ok, self.text = status, payload, status < 400, str(payload)
    def json(self):
        return self._p

good = Resp(200, {"candidates": [{"content": {"parts": [{"text": '```json\n{"is_food": true, "items": []}\n```'}]}}]})
with mock.patch("requests.post", return_value=good) as post:
    v = gemini.make_vlm(api_key="K", model="m")
    out = v(img, T.classes)
    check("JSON berpagar ``` terbaca", out == {"is_food": True, "items": []})
    check("API key lewat header, bukan URL", "x-goog-api-key" in post.call_args.kwargs["headers"]
          and "K" not in post.call_args.args[0])
with mock.patch("requests.post", return_value=Resp(400, "bad")), mock.patch("time.sleep"):
    try:
        gemini.make_vlm(api_key="K")(img, T.classes)
        check("HTTP 400 -> GeminiError", False)
    except gemini.GeminiError:
        check("HTTP 400 -> GeminiError", True)
with mock.patch("requests.post", side_effect=[Resp(503, "x"), good]), mock.patch("time.sleep"):
    check("503 di-retry lalu sukses", gemini.make_vlm(api_key="K")(img, T.classes)["is_food"] is True)
with mock.patch("requests.post", return_value=Resp(200, {"promptFeedback": {}})):
    try:
        gemini.make_vlm(api_key="K")(img, T.classes)
        check("respons tanpa kandidat -> GeminiError", False)
    except gemini.GeminiError:
        check("respons tanpa kandidat -> GeminiError", True)
with mock.patch.dict("os.environ", {}, clear=True), mock.patch("gscv.gemini.load_dotenv"):
    check("tanpa API key -> None", gemini.make_vlm() is None)

print(f"\nPASS: {OK}   FAIL: {FAIL}")
sys.exit(1 if FAIL else 0)
