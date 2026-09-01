"""FAISS visual search runtime for vector queries, loaded once at startup.

The CLIP embedding model runs in the client (transformers.js in the browser).
This backend only needs the prebuilt FAISS index (IndexFlatL2) plus the
product records: it accepts a 512-d embedding vector and returns top-K
visually-similar products using plain Euclidean distance.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np

from . import config

logger = logging.getLogger("foodguard.visualsearch")

FLOAT_DTYPE = np.float32


class SearchRuntime:
    """Loads the FAISS index and records once, then serves vector searches."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._records: list[dict] = []

    @property
    def ready(self) -> bool:
        return self._index is not None

    def load(self) -> None:
        for name, path in [
            ("FAISS", config.FAISS_FILE),
            ("FEATURES", config.FEATURES_FILE),
            ("LAYER1", config.LAYER1_FILE),
        ]:
            if not path.is_file():
                raise RuntimeError(f"Missing visual search file '{path.name}' at: {path}")
            logger.info("Found %s (%s MB)", name, round(path.stat().st_size / (1024 * 1024), 2))

        logger.info("Loading FAISS index from %s", config.FAISS_FILE)
        self._index = faiss.read_index(str(config.FAISS_FILE))
        logger.info("Loaded %s vectors (dim %s)", self._index.ntotal, self._index.d)

        with open(config.FEATURES_FILE, "r", encoding="utf-8") as f:
            self._records = list(json.load(f))
        logger.info("Loaded %s product records", len(self._records))

        if self._index.d != config.EMBED_DIM:
            raise RuntimeError(
                f"FAISS index dim {self._index.d} != expected {config.EMBED_DIM}"
            )

    # ── Product names ──────────────────────────────────────────────────────

    @staticmethod
    def product_name(rec: dict) -> str:
        pid = str(rec.get("product_id", ""))
        if pid:
            return re.sub(r"_\d+$", "", pid)
        img_p = str(rec.get("image_path", "")).replace("\\", "/")
        parts = img_p.split("/")
        return parts[-2] if len(parts) >= 2 else "Unknown Product"

    # ── Search ─────────────────────────────────────────────────────────────

    def search_vector(self, vector, top_k: int = 5) -> list[dict]:
        """Return top-K products for a raw 512-d embedding vector.

        `vector` must be a sequence of length EMBED_DIM containing finite
        numbers. It is used directly (NOT L2-normalized) to match the raw
        open_clip outputs stored in the IndexFlatL2 index, matching Euclidean
        distance semantics.
        """
        if not self.ready:
            raise RuntimeError("visual search runtime not loaded")

        arr = np.asarray(vector, dtype=FLOAT_DTYPE)
        if arr.shape != (config.EMBED_DIM,):
            raise ValueError(
                f"expected embedding of length {config.EMBED_DIM}, got shape {arr.shape}"
            )
        query = np.ascontiguousarray(arr.reshape(1, -1))

        distances, indices = self._index.search(query, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if int(idx) < 0:
                continue
            rec = self._records[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "product_name": self.product_name(rec),
                    "product_id": rec.get("product_id"),
                    "score": float(score),
                    "image_path": rec.get("image_path"),
                }
            )
        return results


_runtime: SearchRuntime | None = None


def get_runtime() -> SearchRuntime:
    global _runtime
    if _runtime is None:
        _runtime = SearchRuntime()
    return _runtime
