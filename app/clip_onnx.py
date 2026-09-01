"""
Server-side CLIP ViT-B/32 image embedding via ONNX Runtime.

Runs the vision tower in ONNX (no torch / open_clip / transformers). The model
is loaded ONCE at startup and cached; it is never re-downloaded per request.

Output is the RAW (non-normalized) 512-d CLIP image embedding, matching the
embeddings stored in the backend's IndexFlatL2 (Euclidean) index. Preprocessing
mirrors open_clip ViT-B-32 "openai" exactly (RGB -> resize shortest side to 224
bicubic -> centre-crop 224 -> normalize with the CLIP openai mean/std).
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from . import config

logger = logging.getLogger("foodguard.visualsearch")

# CLIP ViT-B-32 "openai" normalization constants (identical to the reference
# build in search/run.py and the model's preprocessor_config.json).
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

IMG_SIZE = 224


def _resolve_model_path() -> Path:
    """Locate the ONNX vision model (env override, then repo runtime dir)."""
    env = os.environ.get("CLIP_ONNX_MODEL")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"CLIP_ONNX_MODEL set but not found: {p}")

    # Preference: a dedicated runtime dir, then the in-repo search/ assets,
    # then the search/ quantized variant.
    candidates = [
        config.BASE_DIR / "clip_visual_quantized.onnx",
        config.BASE_DIR / "clip_visual.onnx",
        config.PROJECT_ROOT / "search" / "clip_visual_quantized.onnx",
        config.PROJECT_ROOT / "search" / "clip_visual.onnx",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "No CLIP ONNX model found. Set CLIP_ONNX_MODEL or place "
        "clip_visual_quantized.onnx next to the app."
    )


class CLIPONNXRuntime:
    """Loads the CLIP vision ONNX session once and embeds images to 512-d."""

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._model_path: Path | None = None

    @property
    def ready(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        model_path = _resolve_model_path()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

        in_shape = inputs[0].shape
        out_shape = outputs[0].shape
        logger.info(
            "ONNX CLIP loaded from %s: input=%s output=%s", model_path, in_shape, out_shape
        )
        if len(in_shape) >= 2 and in_shape[-3:] != [3, IMG_SIZE, IMG_SIZE] and in_shape[-2:] != [IMG_SIZE, IMG_SIZE]:
            raise RuntimeError(f"Unexpected CLIP input shape {in_shape}")
        if out_shape[-1] != config.EMBED_DIM:
            raise RuntimeError(
                f"ONNX CLIP output dim {out_shape} != expected {config.EMBED_DIM}"
            )

        self._session = session
        self._model_path = model_path

    def embed_image_bytes(self, data: bytes) -> np.ndarray:
        """Embed raw image bytes into a raw 512-d (non-normalized) CLIP vector."""
        if not self.ready:
            raise RuntimeError("CLIP ONNX runtime not loaded")
        image = Image.open(io.BytesIO(data))
        tensor = self._preprocess(image)
        outputs = self._session.run([self._output_name], {self._input_name: tensor})
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1).copy()

    @staticmethod
    def _preprocess(img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")
        w, h = img.size
        if w < h:
            new_w, new_h = IMG_SIZE, int(round(h * (IMG_SIZE / w)))
        else:
            new_w, new_h = int(round(w * (IMG_SIZE / h))), IMG_SIZE
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)
        left = (new_w - IMG_SIZE) // 2
        top = (new_h - IMG_SIZE) // 2
        img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - CLIP_MEAN) / CLIP_STD
        return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0).astype(np.float32).copy()


_clip: CLIPONNXRuntime | None = None


def get_clip() -> CLIPONNXRuntime:
    global _clip
    if _clip is None:
        _clip = CLIPONNXRuntime()
    return _clip
