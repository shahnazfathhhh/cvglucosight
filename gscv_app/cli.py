"""Uji cepat di terminal:  python cli.py foto.jpg   (tambah --local untuk paksa model lokal)"""
import argparse
import json

from gscv.config import MODEL_PATH, TABLE_PATH
from gscv.gemini import make_vlm
from gscv.local import make_local
from gscv.pipeline import analyze, format_result, to_cv_record
from gscv.table import FoodTable

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--local", action="store_true", help="jangan pakai Gemini")
ap.add_argument("--json", action="store_true", help="cetak record untuk Forecasting")
args = ap.parse_args()

table = FoodTable(TABLE_PATH)
res = analyze(args.image, table, vlm=None if args.local else make_vlm(), local=make_local(MODEL_PATH))
print(json.dumps(to_cv_record(res), indent=2, ensure_ascii=False) if args.json else format_result(res))
