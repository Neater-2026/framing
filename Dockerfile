# Render 極輕量 CPU 專用 PyTorch Dockerfile (徹底防止 512MB RAM 超載)
FROM python:3.10-slim

WORKDIR /app

# 設定記憶體限制與線程優化
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PORT=10000

# 安裝極輕量 CPU-only PyTorch (不包含 1.5GB 的 NVIDIA CUDA 共享庫，記憶體直接省下 70%)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir numpy fastapi uvicorn pydantic

COPY . .

RUN mkdir -p data/replay_buffer checkpoints

EXPOSE 10000

CMD ["python", "run_ai_service.py"]
