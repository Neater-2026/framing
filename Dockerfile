# Render 記憶體優化版 Dockerfile (避免 512MB RAM OOM Status 137 錯誤)
FROM python:3.10-slim

WORKDIR /app

# 設定記憶體限制優化環境變數，防止 PyTorch 多線程吃滿 512MB 記憶體
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PORT=10000

# 複製並安裝相依套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製 server 所有原始碼
COPY . .

RUN mkdir -p data/replay_buffer checkpoints

EXPOSE 10000

CMD ["python", "run_ai_service.py"]
