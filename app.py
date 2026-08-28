import os
import json
import time
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any

from framing_env import FramingEnv
from model import FramingAlphaNet
from mcts import MCTS

app = FastAPI(title="Framing AI Engine Server", version="1.0.0")

# 允許跨域請求 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'checkpoints/champion.pt'

# 載入當前 Champion 模型
model = FramingAlphaNet().to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"FastAPI API 伺服器已成功載入王者模型: {model_path}")
else:
    print("未檢測到模型檔，使用預設初始化 AlphaZero 網路。")
model.eval()

# Pydantic 數據模型
class CellState(BaseModel):
    player: int
    value: int

class PlayerState(BaseModel):
    score: int
    hand: List[int]

class PredictRequest(BaseModel):
    board: List[Optional[CellState]]
    players: List[PlayerState]
    currentTurn: int
    simulations: Optional[int] = 50

class SocialLoginRequest(BaseModel):
    provider: str
    providerId: str
    username: str
    email: Optional[str] = None
    avatarUrl: Optional[str] = None

from auth import get_or_create_social_user

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Framing AlphaZero AI Server"}

@app.post("/api/auth/social-login")
def social_login(req: SocialLoginRequest):
    try:
        user = get_or_create_social_user(
            provider=req.provider,
            provider_id=req.providerId,
            username=req.username,
            email=req.email,
            avatar_url=req.avatarUrl
        )
        return {"status": "success", "user": user}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/version")
def get_model_version():
    meta_path = 'checkpoints/metadata.json'
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "version": "1.0.0-default",
        "elo": 1000,
        "model_name": "FramingAlphaZero-v1"
    }

@app.post("/api/predict")
def predict_action(req: PredictRequest):
    try:
        env = FramingEnv()
        # 轉換前端傳入的盤面與手牌
        env.board = [
            {'player': c.player, 'value': c.value} if c is not None else None
            for c in req.board
        ]
        env.players = [
            {'score': p.score, 'hand': list(p.hand)}
            for p in req.players
        ]
        env.current_turn = req.currentTurn

        # 執行 MCTS 搜尋
        sims = req.simulations or 50
        mcts = MCTS(model, n_simulations=sims, device=device)
        action_probs, expected_values = mcts.search(env)

        best_action_id = int(np.argmax(action_probs))
        cell_index = best_action_id // 9
        value = (best_action_id % 9) + 1

        # 取前 3 個熱門推薦動作
        top_indices = np.argsort(action_probs)[::-1][:3]
        top_candidates = []
        for idx in top_indices:
            if action_probs[idx] > 0:
                top_candidates.append({
                    'cellIndex': int(idx // 9),
                    'value': int((idx % 9) + 1),
                    'probability': round(float(action_probs[idx]), 4)
                })

        return {
            'bestAction': {
                'cellIndex': cell_index,
                'value': value,
                'actionId': best_action_id
            },
            'winrates': [round(float(v), 4) for v in expected_values],
            'topCandidates': top_candidates
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/replay")
def submit_replay(req: ReplayLogRequest):
    os.makedirs('data/replay_buffer', exist_ok=True)
    filename = f'data/replay_buffer/user_game_{int(time.time()*1000)}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(req.dict(), f, indent=2)
    return {"status": "success", "message": "棋譜已接收並加入訓練庫"}

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"FastAPI 伺服器啟動於 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
