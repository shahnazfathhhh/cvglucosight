"""Alur: foto -> (Gemini | MobileNet lokal) -> item + gram -> tabel per 100 g -> makro.

Setiap hasil bisa dikoreksi user lewat from_rows() (kelas dan gram manual).
"""
import math

from .config import MANUAL_CONF_THRESHOLD, MAX_GRAMS, MIN_GRAMS


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clamp(grams, label, warnings):
    if grams < MIN_GRAMS or grams > MAX_GRAMS:
        new = min(max(grams, MIN_GRAMS), MAX_GRAMS)
        warnings.append(f"gram '{label}' ({grams:g}) di luar {MIN_GRAMS:g}-{MAX_GRAMS:g} g, dipotong ke {new:g}")
        return new
    return grams


def _item(label, food_class, grams, class_source, grams_source, confidence, low=None, high=None):
    return {"label": label, "food_class": food_class, "grams": round(grams, 1),
            "grams_low": low, "grams_high": high, "class_source": class_source,
            "grams_source": grams_source, "confidence": confidence}


def parse_vlm(raw, table):
    """Respons Gemini -> (items, is_food, warnings). Raise ValueError kalau bentuknya salah."""
    if not isinstance(raw, dict):
        raise ValueError("respons VLM bukan objek JSON")
    warnings, items = [], []
    if raw.get("is_food") is False:
        return [], False, warnings
    for it in raw.get("items") or []:
        label = str(it.get("name") or "makanan").strip()
        grams = _num(it.get("grams"))
        if grams is None:
            warnings.append(f"'{label}' dilewati: gram tidak valid")
            continue
        cls = it.get("table_class")
        cls = cls if isinstance(cls, str) and table.has(cls) else None
        conf = min(max(_num(it.get("confidence")) or 0.0, 0.0), 1.0)
        items.append(_item(label, cls, _clamp(grams, label, warnings), "vlm", "vlm_estimate",
                           conf, _num(it.get("grams_low")), _num(it.get("grams_high"))))
    return items, True, warnings


def compute(items, table):
    """Isi macros per item dan hitung total. Item tanpa kelas di tabel tidak masuk total."""
    totals = {"carbs_g": 0.0, "protein_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0, "kcal": 0}
    complete = True
    for it in items:
        if it["food_class"] and table.has(it["food_class"]):
            it["macros"] = table.macros(it["food_class"], it["grams"])
            for k in totals:
                totals[k] += it["macros"][k]
        else:
            it["macros"], complete = None, False
    return {k: round(v, 1) if k != "kcal" else int(v) for k, v in totals.items()}, complete


def _finish(engine, items, is_food, warnings, table):
    totals, complete = compute(items, table)
    reasons = []
    if not is_food:
        reasons.append("foto tidak terdeteksi sebagai makanan")
    elif not items:
        reasons.append("tidak ada makanan terdeteksi")
    for it in items:
        if it["macros"] is None:
            reasons.append(f"'{it['label']}' tidak ada di tabel makro: pilih kelas terdekat atau hapus barisnya")
        elif it["class_source"] != "user" and it["confidence"] < MANUAL_CONF_THRESHOLD:
            reasons.append(f"keyakinan rendah ({it['confidence']:.2f}) untuk '{it['label']}': periksa kelas dan gram")
    return {"engine": engine, "items": items, "totals": totals, "total_complete": complete,
            "needs_manual_input": bool(reasons), "reasons": reasons, "warnings": warnings}


def analyze(image_path, table, vlm=None, local=None):
    """Coba Gemini dulu; kalau tidak ada/gagal, jatuh ke MobileNet lokal; kalau itu juga tidak ada, minta manual."""
    warnings = []
    if vlm is not None:
        try:
            items, is_food, w = parse_vlm(vlm(image_path, table.classes), table)
            return _finish("gemini", items, is_food, warnings + w, table)
        except Exception as exc:  # GeminiError, ValueError, dll.
            warnings.append(f"Gemini gagal ({exc}); memakai model lokal")
    else:
        warnings.append("GEMINI_API_KEY tidak diset; memakai model lokal")

    if local is not None:
        try:
            cls, conf = local.predict_topk(image_path, k=1)[0]
            items = [_item(table.rows[cls]["name"], cls, table.rows[cls]["serving_g"],
                           "local", "class_default", round(conf, 4))]
            warnings.append("model lokal hanya mengenali kelas; gram = porsi standar kelas, BUKAN estimasi dari foto")
            return _finish("local", items, True, warnings, table)
        except Exception as exc:
            warnings.append(f"model lokal gagal ({exc})")
    return _finish("none", [], True, warnings + ["tidak ada mesin yang bisa menganalisis foto"], table)


def from_rows(rows, table):
    """Hitung ulang dari baris yang diedit user: [[nama, kelas, gram], ...]."""
    items, warnings = [], []
    for row in rows:
        cells = list(row) + [None] * 3
        label = str(cells[0] or "").strip()
        cls_raw = str(cells[1] or "").strip().lower().replace(" ", "_")
        grams = _num(cells[2])
        if not label and not cls_raw and grams is None:
            continue  # baris kosong
        label = label or cls_raw or "makanan"
        if grams is None:
            warnings.append(f"'{label}' dilewati: gram kosong atau bukan angka")
            continue
        cls = cls_raw if table.has(cls_raw) else None
        items.append(_item(label, cls, _clamp(grams, label, warnings), "user", "user_manual", 1.0))
    return _finish("manual", items, True, warnings, table)


def to_cv_record(result):
    """Ringkasan untuk Forecasting. NILAI carbs_source DI SINI BARU: perlu persetujuan PM (kontrak v1.2)."""
    srcs = {i["grams_source"] for i in result["items"]}
    source = ("user_grams_x_table" if "user_manual" in srcs else
              "vlm_grams_x_table" if "vlm_estimate" in srcs else "class_lookup")
    t = result["totals"]
    return {"carbs_g": t["carbs_g"], "protein_g": t["protein_g"], "fat_g": t["fat_g"],
            "fiber_g": t["fiber_g"], "calories_kcal": t["kcal"],
            "cv_present": int(bool(result["items"])), "carbs_source": source,
            "total_complete": result["total_complete"], "engine": result["engine"],
            "fiber_source": "estimated_from_carb_ratio"}


def format_result(result):
    lines = [f"Mesin: {result['engine']}", "",
             "| Item | Kelas | Gram | Karbo | Protein | Lemak | Serat* | kkal |", "|---|---|---|---|---|---|---|---|"]
    for it in result["items"]:
        g = f"{it['grams']:g}"
        if it["grams_low"] is not None and it["grams_high"] is not None:
            g += f" ({it['grams_low']:g}-{it['grams_high']:g})"
        m = it["macros"]
        vals = [m[k] for k in ("carbs_g", "protein_g", "fat_g", "fiber_g", "kcal")] if m else ["-"] * 5
        lines.append(f"| {it['label']} | {it['food_class'] or '(tidak ada di tabel)'} | {g} | " +
                     " | ".join(str(v) for v in vals) + " |")
    t = result["totals"]
    lines += ["", f"**Total:** karbo {t['carbs_g']} g, protein {t['protein_g']} g, lemak {t['fat_g']} g, "
                  f"serat* {t['fiber_g']} g, {t['kcal']} kkal"]
    if not result["total_complete"]:
        lines.append("**Total belum lengkap:** ada item yang tidak masuk hitungan.")
    for r in result["reasons"]:
        lines.append(f"- Perlu dicek: {r}")
    for w in result["warnings"]:
        lines.append(f"- Catatan: {w}")
    lines += ["", "*serat adalah heuristik dari rasio karbo, bukan data TKPI. Gram dari foto adalah estimasi kasar. "
                  "Prototipe riset, bukan alat medis."]
    return "\n".join(lines)
