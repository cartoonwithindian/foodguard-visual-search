import math
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Form, HTTPException, UploadFile

from .. import config
from ..auth import require_api_key
from ..clip_onnx import get_clip
from ..search_runtime import get_runtime

router = APIRouter(prefix="/api/v1")


def _validate_image_type(mime: str) -> None:
    allowed = ("image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "image/bmp", "image/gif")
    if mime.split(";")[0].strip().lower() not in allowed:
        raise HTTPException(status_code=422, detail=f"Unsupported image type: {mime}")


@router.post(
    "/search_by_vector",
    summary="Top-K visually similar products for an embedding vector",
    dependencies=[Depends(require_api_key)],
)
def search_by_vector(
    payload: Annotated[
        dict[str, Any],
        Body(
            ...,
            description=(
                'JSON body: {"vector": [512 floats], "top_k": 5}. The vector is '
                "a raw (non-normalized) CLIP ViT-B-32 image embedding produced "
                "client-side."
            ),
            examples=[{"vector": [0.0] * config.EMBED_DIM, "top_k": 5}],
        ),
    ],
):
    vector = payload.get("vector")
    if not isinstance(vector, (list, tuple)):
        raise HTTPException(status_code=422, detail="'vector' must be an array of numbers")

    if len(vector) != config.EMBED_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"'vector' must contain exactly {config.EMBED_DIM} numbers "
            f"(got {len(vector)})",
        )

    # All values must be finite real numbers.
    try:
        for v in vector:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ValueError
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="'vector' must contain only finite numeric values",
        ) from None

    top_k = payload.get("top_k", 5)
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="'top_k' must be an integer") from None
    if top_k < 1:
        raise HTTPException(status_code=422, detail="'top_k' must be >= 1")

    rt = get_runtime()
    try:
        results = rt.search_vector(vector, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # runtime not loaded / index errors
        raise HTTPException(status_code=503, detail=f"Search unavailable: {exc}") from exc

    return {"query": "vector_search", "results": results}


@router.post(
    "/search",
    summary="Top-K visually similar products for an uploaded image",
    dependencies=[Depends(require_api_key)],
)
async def search_image(
    image: UploadFile,
    top_k: Annotated[int, Form()] = 5,
):
    """Upload an image and return top-K visually-similar products.

    The image is embedded server-side with an ONNX CLIP ViT-B/32 model and
    searched against the same IndexFlatL2 index used by search_by_vector.
    """
    if top_k < 1:
        raise HTTPException(status_code=422, detail="'top_k' must be >= 1")

    _validate_image_type(image.content_type or "application/octet-stream")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded image is empty")

    clip = get_clip()
    if not clip.ready:
        raise HTTPException(status_code=503, detail="CLIP model not loaded")

    try:
        vector = clip.embed_image_bytes(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Unable to process image: {exc}") from exc

    rt = get_runtime()
    try:
        results = rt.search_vector(vector, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search unavailable: {exc}") from exc

    return {"query": "image_search", "results": results}
