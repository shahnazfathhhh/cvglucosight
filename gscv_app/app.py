"""Demo web (Gradio):  python app.py  lalu buka http://127.0.0.1:7860"""
import gradio as gr

from gscv.config import MODEL_PATH, TABLE_PATH
from gscv.gemini import make_vlm
from gscv.local import make_local
from gscv.pipeline import analyze, format_result, from_rows
from gscv.table import FoodTable

TABLE = FoodTable(TABLE_PATH)
VLM = make_vlm()
LOCAL = make_local(MODEL_PATH)
HEADERS = ["Nama", "Kelas tabel", "Gram"]


def _rows(result):
    return [[i["label"], i["food_class"] or "", i["grams"]] for i in result["items"]]


def on_analyze(image_path):
    if not image_path:
        return [], "Upload foto dulu."
    res = analyze(image_path, TABLE, vlm=VLM, local=LOCAL)
    return _rows(res), format_result(res)


def on_recalc(rows):
    return format_result(from_rows(rows, TABLE))


with gr.Blocks(title="GlucoSight CV") as demo:
    gr.Markdown("# GlucoSight CV: foto makanan ke gram dan makro\n"
                f"Mesin utama: **{'Gemini' if VLM else 'model lokal (GEMINI_API_KEY belum diset)'}**. "
                "Edit tabel (nama, kelas, gram) lalu klik *Hitung ulang* jika hasilnya salah.")
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="filepath", label="Foto makanan")
            btn = gr.Button("Analisis", variant="primary")
        with gr.Column():
            gr.Markdown("Kelas tabel yang valid: `" + "`, `".join(TABLE.classes) + "`")
            rows = gr.Dataframe(headers=HEADERS, datatype=["str", "str", "number"],
                                interactive=True, row_count=(1, "dynamic"), col_count=(3, "fixed"),
                                type="array", label="Item (bisa diedit, bisa tambah baris)")
            btn2 = gr.Button("Hitung ulang dari tabel di atas")
            out = gr.Markdown()
    btn.click(on_analyze, img, [rows, out])
    btn2.click(on_recalc, rows, out)

if __name__ == "__main__":
    demo.launch()
