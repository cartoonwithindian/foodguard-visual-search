---
title: FoodGuard Visual Search API
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# FoodGuard Visual Search API

Production FastAPI service for FoodGuard's visual product search.

- **AI model:** CLIP ViT-B/32 (ONNX, `clip_visual_quantized.onnx`, INT8 ~85 MB)
- **Search:** FAISS `IndexFlatL2` over precomputed raw (non-normalized) image
  embeddings; aligns with the raw CLIP embeddings that built the index.
- **Runtime:** ONNX Runtime (CPU) + FAISS — no torch / no open_clip / no GPU.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Lightweight health + readiness (no inference) |
| GET  | `/docs` | Swagger UI |
| POST | `/api/v1/search_by_vector` | JSON `{vector:[512 floats], top_k}` |
| POST | `/api/v1/search` | `multipart/form-data` `image` (UploadFile) + `top_k` (Form) |

Both search endpoints return the same shape:

```json
{
  "query": "vector_search" | "image_search",
  "results": [
    {"rank": 1, "product_name": "...", "product_id": "...", "score": 0.0, "image_path": "..."}
  ]
}
```

## Deploy as a Hugging Face Docker Space

This directory is a Docker Space (`sdk: docker`). Hugging Face builds the
`Dockerfile`, passes the `PORT` env var, and the app binds `0.0.0.0`.

### Required runtime assets

The image downloads all four runtime assets **once at build time** from the
project assets repo `nazimtovo/foodguard-assets` (no network dependency at
boot, image is self-contained):

| Asset | Expected filename | Used by |
|-------|-------------------|---------|
| CLIP model | `clip_visual_quantized.onnx` (INT8 ~85 MB) | `/api/v1/search` |
| FAISS index | `products_images_faiss_index_v2.bin` | vector + image search |
| Product records | `products_images_features_v2.json` | result metadata |
| Layer1 quality | `products_layer1_quality.json` | (bundled metadata) |

These four files **must exist in `nazimtovo/foodguard-assets` on Hugging Face
before building** — the `clip_visual_quantized.onnx` wget step is a hard build
step and fails the build if the file is missing (the model is required at
runtime). Do not commit the ~335 MB fp32 `clip_visual.onnx` to the Space.

### Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PORT` | Listen port (Hugging Face sets this) | `8001` |
| `CLIP_ONNX_MODEL` | Path to the ONNX model | `/srv/app/assets/clip_visual_quantized.onnx` |
| `FOODGUARD_RUNTIME` | Dir containing FAISS index + metadata | `/srv/app/assets` |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (empty = CORS disabled) | `""` |
| `FOODGUARD_API_KEY` | Optional bearer auth; empty = no auth | unset |

> CORS is only enabled when `ALLOWED_ORIGINS` is non-empty. It never falls back
> to `*`.

### Local run

```bash
# from this repo root, using the bundled model + assets
CLIP_ONNX_MODEL=search/clip_visual_quantized.onnx \
FOODGUARD_RUNTIME=search \
uvicorn app.main:app --host 0.0.0.0 --port 8001
```
