import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

APP_NAME = "FoodGuard Visual Product Search API"
API_VERSION = "1.0.0"

APP_DESCRIPTION = (
    "Read-only FAISS visual product search service. Accepts a raw image "
    "embedding vector (512-d, produced client-side by CLIP ViT-B-32 via "
    "transformers.js) and returns top-K visually-similar products from the "
    "FoodGuard product image index (13,671 vectors). The FAISS index and "
    "records are loaded once at startup. Used by the main FoodGuard app as a "
    "fallback when a scanned barcode cannot be matched to a product."
)

# Dimensionality of the CLIP ViT-B-32 (openai) image embeddings stored in the
# FAISS index. Must match the client-side transformers.js model output.
EMBED_DIM = 512

# Optional production API key. When empty, authentication is disabled (local dev).
API_KEY = (os.environ.get("FOODGUARD_API_KEY") or "").strip() or None

# Resolve the runtime directory containing the FAISS index / features.
# FOODGUARD_RUNTIME env may point at it; otherwise fall back to the bundled
# `search/` runtime that ships with the repo.
def _resolve_runtime_dir() -> Path:
    env = os.environ.get("FOODGUARD_RUNTIME")
    if env:
        return Path(env)
    script_dir = BASE_DIR
    if (script_dir / "products_images_faiss_index_v2.bin").exists():
        return script_dir
    search_dir = PROJECT_ROOT / "search"
    if (search_dir / "products_images_faiss_index_v2.bin").exists():
        return search_dir
    if (search_dir / "FOODGUARD_RUNTIME" / "products_images_faiss_index_v2.bin").exists():
        return search_dir / "FOODGUARD_RUNTIME"
    return search_dir

RUNTIME_DIR = _resolve_runtime_dir()

FAISS_FILE = RUNTIME_DIR / "products_images_faiss_index_v2.bin"
FEATURES_FILE = RUNTIME_DIR / "products_images_features_v2.json"
LAYER1_FILE = RUNTIME_DIR / "products_layer1_quality.json"

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]

