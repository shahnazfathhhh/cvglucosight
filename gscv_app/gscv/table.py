"""Tabel makro per 100 g. Makro = nilai_per_100g * gram / 100."""
import csv
from pathlib import Path


class FoodTable:
    def __init__(self, path):
        self.rows = {}
        with open(Path(path), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows[r["food_class"]] = {
                    "name": r["tkpi_item_name"],
                    "serving_g": float(r["serving_g"]),
                    "kcal": float(r["energy_kcal_100g"]),
                    "carbs": float(r["carbs_g_100g"]),
                    "protein": float(r["protein_g_100g"]),
                    "fat": float(r["fat_g_100g"]),
                    "fiber_ratio": float(r["fiber_ratio_of_carbs"]),  # heuristik, bukan TKPI
                    "mapping_confidence": r["mapping_confidence"],
                }
        if not self.rows:
            raise ValueError("Tabel makro kosong")

    @property
    def classes(self):
        return sorted(self.rows)

    def has(self, food_class):
        return food_class in self.rows

    def macros(self, food_class, grams):
        r, k = self.rows[food_class], grams / 100.0
        carbs = r["carbs"] * k
        return {
            "carbs_g": round(carbs, 1),
            "protein_g": round(r["protein"] * k, 1),
            "fat_g": round(r["fat"] * k, 1),
            "fiber_g": round(carbs * r["fiber_ratio"], 1),
            "kcal": round(r["kcal"] * k),
        }
