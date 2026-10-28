from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE_PATH = ROOT / "data" / "food_table.csv"
MODEL_PATH = ROOT / "models" / "mobilenetv2_food10.pt"

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"  # sama dengan eksperimen vlm_vs_mobilenet
MIN_GRAMS, MAX_GRAMS = 10.0, 1000.0
# Di bawah ini app meminta user memeriksa/mengisi manual. Confidence VLM dan
# softmax belum terkalibrasi, jadi angka ini hanya titik awal.
MANUAL_CONF_THRESHOLD = 0.50
