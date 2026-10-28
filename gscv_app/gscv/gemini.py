"""Klien Gemini (REST) -> daftar item makanan + estimasi gram dari foto."""
import base64
import io
import json
import os
import time

import requests
from dotenv import load_dotenv
from PIL import Image

from .config import DEFAULT_GEMINI_MODEL

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"


class GeminiError(Exception):
    pass


PROMPT = """You are a nutrition assistant for photos of Indonesian meals.
List each distinct dish or component visible in the photo. For each item give:
- name: common Indonesian name
- table_class: exactly one of [{classes}] if the item clearly is that dish, otherwise "other"
- grams: your best estimate of the weight in grams of that item as served (food only, no plate; include broth for soups)
- grams_low, grams_high: a plausible range for grams
- confidence: 0.0-1.0 for how sure you are of the identification
Use plate/bowl size, spoon, fork and hands as scale cues.
If the photo does not show food, return is_food=false and an empty items list.
Return JSON only, no markdown:
{{"is_food": true, "items": [{{"name": "...", "table_class": "...", "grams": 0, "grams_low": 0, "grams_high": 0, "confidence": 0.0}}]}}"""


def _jpeg_b64(image_path, max_side=1024):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_vlm(api_key=None, model=None, timeout=60, retries=2):
    """Kembalikan fungsi vlm(image_path, classes) -> dict, atau None kalau tidak ada API key."""
    load_dotenv()
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    def vlm(image_path, classes):
        body = {
            "contents": [{"parts": [
                {"text": PROMPT.format(classes=", ".join(classes))},
                {"inline_data": {"mime_type": "image/jpeg", "data": _jpeg_b64(image_path)}},
            ]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            },
        }
        url = f"{BASE_URL}{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        resp = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            except requests.RequestException as exc:
                if attempt == retries:
                    raise GeminiError(f"koneksi gagal: {type(exc).__name__}") from exc
                time.sleep(2 ** attempt)
                continue
            if resp.ok:
                break
            if resp.status_code in (429, 500, 503) and attempt < retries:
                time.sleep(2 ** attempt * 2)
                continue
            raise GeminiError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            parts = resp.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(text)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise GeminiError("respons Gemini kosong atau bukan JSON valid") from exc

    return vlm
