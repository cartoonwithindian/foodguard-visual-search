from fastapi import APIRouter

from .. import config
from ..clip_onnx import get_clip
from ..search_runtime import get_runtime

router = APIRouter()


@router.get("/health", summary="Health + runtime availability (no auth required)")
def health():
    rt = get_runtime()
    return {
        "status": "ok",
        "service": config.APP_NAME,
        "api_version": config.API_VERSION,
        "runtime_ready": rt.ready,
        "model_ready": get_clip().ready,
        "vectors": rt._index.ntotal if rt.ready else None,
    }
