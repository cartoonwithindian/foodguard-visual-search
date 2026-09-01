import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .clip_onnx import get_clip
from .routers import health, search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("foodguard.visualsearch")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    rt = search.get_runtime()
    try:
        rt.load()
    except Exception as exc:  # missing assets / model download failure
        logger.error("Visual search runtime failed to load: %s", exc)
        # Keep serving /health with runtime_ready=false so the app can degrade
        # gracefully instead of crashing on startup.
    # ONNX CLIP is required for server-side image search. If it cannot load,
    # fail startup with a clear error rather than serving an incompatible model.
    get_clip().load()
    yield


app = FastAPI(
    title=config.APP_NAME,
    version=config.API_VERSION,
    description=config.APP_DESCRIPTION,
    lifespan=lifespan,
)

if config.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

app.include_router(health.router)
app.include_router(search.router)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    content = {"error": {"code": "HTTP_ERROR", "message": str(exc.detail or exc.status_code)}}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc: RequestValidationError):
    first = (exc.errors() or [{}])[0]
    loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
    msg = f"Invalid request: {first.get('msg', 'validation error')}" + (f" at '{loc}'" if loc else "")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": {"code": "INVALID_INPUT", "message": msg}},
    )
