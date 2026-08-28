# Hugging Face Spaces & Cloud Dockerfile for Framing AlphaZero AI Server
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統與編譯工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 相依套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製 server 所有原始碼
COPY . .

# 建立數據與 Checkpoint 目錄
RUN mkdir -p data/replay_buffer checkpoints

# 暴露 Hugging Face 預設 Port 7860
EXPOSE 7860

# 啟動 AI 服務主入口 (支援背景不間斷學習與 FastAPI 推論)
CMD ["python", "run_ai_service.py"]
