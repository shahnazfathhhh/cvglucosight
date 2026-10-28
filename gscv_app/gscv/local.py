"""Fallback lokal: MobileNetV2 (hanya klasifikasi, tanpa estimasi gram). torch di-import lazy."""
from pathlib import Path

import numpy as np
from PIL import Image

MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


class LocalUnavailable(Exception):
    pass


def _preprocess(path):
    """Resize sisi pendek 255 -> center crop 224 -> normalisasi ImageNet (sama dengan training)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    nw, nh = (255, round(h * 255 / w)) if w < h else (round(w * 255 / h), 255)
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = round((nw - 224) / 2), round((nh - 224) / 2)
    img = img.crop((left, top, left + 224, top + 224))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1).astype(np.float32)


class LocalClassifier:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self._model = None

    def _load(self):
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
        except ImportError as exc:
            raise LocalUnavailable("torch/torchvision belum terpasang") from exc
        ckpt = torch.load(self.model_path, map_location="cpu")
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, ckpt["num_classes"])
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        self._torch, self._model = torch, m
        self._idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}

    def predict_topk(self, image_path, k=3):
        if self._model is None:
            self._load()
        x = self._torch.from_numpy(_preprocess(image_path)).unsqueeze(0)
        with self._torch.no_grad():
            logits = self._model(x)[0].numpy()
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        order = np.argsort(probs)[::-1][:k]
        return [(self._idx_to_class[int(i)], float(probs[i])) for i in order]


def make_local(model_path):
    p = Path(model_path)
    return LocalClassifier(p) if p.exists() else None
