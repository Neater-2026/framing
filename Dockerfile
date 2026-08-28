# Render 極輕量 CPU 專用 PyTorch Dockerfile (修復 PyPI 索引連結)
FROM python:3.10-slim

WORKDIR /app

# 設定記憶體限制與線程優化
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PORT=10000

# 使用 --extra-index-url 保持 PyPI 與 PyTorch CPU 雙源索引，順利安裝輕量版 PyTorch
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir numpy fastapi uvicorn pydantic

COPY . .

RUN mkdir -p data/replay_buffer checkpoints

EXPOSE 10000

CMD ["python", "run_ai_service.py"]
