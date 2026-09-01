FROM python:3.12-slim

WORKDIR /srv/app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FOODGUARD_RUNTIME=/srv/app/assets \
    CLIP_ONNX_MODEL=/srv/app/assets/clip_visual_quantized.onnx \
    ALLOWED_ORIGINS="" \
    PORT=8001

# System libs required by faiss (OpenMP) + Pillow image codecs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so rebuilds reuse the layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the FAISS runtime assets once at build time so the image is
# self-contained and boots without a network round-trip. Assets mirror the
# ones hosted at https://huggingface.co/nazimtovo/foodguard-assets.
RUN mkdir -p /srv/app/assets && \
    wget -q -O /srv/app/assets/products_images_faiss_index_v2.bin \
        https://huggingface.co/nazimtovo/foodguard-assets/resolve/main/products_images_faiss_index_v2.bin && \
    wget -q -O /srv/app/assets/products_images_features_v2.json \
        https://huggingface.co/nazimtovo/foodguard-assets/resolve/main/products_images_features_v2.json && \
    wget -q -O /srv/app/assets/products_layer1_quality.json \
        https://huggingface.co/nazimtovo/foodguard-assets/resolve/main/products_layer1_quality.json && \
    ls -la /srv/app/assets

# Download the CLIP ONNX model used for server-side image search once at build
# time so the image is self-contained. This is a hard build step: the model is
# required (the app fails startup if it is missing), so fail the build here
# rather than shipping a broken image. Hosted at the same project assets repo.
RUN wget -q -O /srv/app/assets/clip_visual_quantized.onnx \
        https://huggingface.co/nazimtovo/foodguard-assets/resolve/main/clip_visual_quantized.onnx && \
    ls -la /srv/app/assets

# Copy the application code.
COPY app ./app

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys,os; p=os.environ.get('PORT','8001'); sys.exit(0) if urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=5).status==200 else sys.exit(1)" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
