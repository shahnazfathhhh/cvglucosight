# GlucoSight CV v2

Foto makanan -> item + estimasi gram (Gemini) -> tabel per 100 g -> makro.
Fallback: MobileNetV2 lokal (hanya kelas, gram = porsi standar). Selalu bisa dikoreksi manual.

## Jalankan di VS Code
1. File > Open Folder > folder ini.
2. Terminal > New Terminal, lalu:
   `python -m venv .venv` dan aktifkan (`.venv\Scripts\activate` di Windows).
3. `pip install -r requirements.txt`
4. Ctrl+Shift+P > "Python: Select Interpreter" > pilih `.venv`.
5. Salin `.env.example` jadi `.env`, isi `GEMINI_API_KEY`.
6. `python tests/test_pipeline.py` (harus 30 PASS, tanpa internet).
7. `python cli.py samples/foto.jpg` untuk uji 1 foto, atau `python app.py` lalu buka http://127.0.0.1:7860.
   Bisa juga lewat Run and Debug (F5) di VS Code.

## Struktur
gscv/table.py (perkalian), gscv/gemini.py (VLM), gscv/local.py (MobileNet), gscv/pipeline.py (alur + koreksi manual).
