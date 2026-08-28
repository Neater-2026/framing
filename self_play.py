import os
import json
import time
import gc
import numpy as np
import torch
from framing_env import FramingEnv
from model import FramingAlphaNet
from mcts import MCTS

class SelfPlayWorker:
    """
    自我對弈與 Replay Buffer 數據生成器 (記憶體優化版)
    """
    def __init__(self, model, buffer_dir='data/replay_buffer', n_simulations=50, device='cpu'):
        self.model = model
        self.buffer_dir = buffer_dir
        self.n_simulations = n_simulations
        self.device = device
        os.makedirs(self.buffer_dir, exist_ok=True)

    @torch.no_grad()
    def run_episode(self):
        env = FramingEnv()
        mcts = MCTS(self.model, n_simulations=self.n_simulations, device=self.device)

        states = []
        policies = []
        current_turns = []

        step_count = 0
        while not env.game_phase == 'gameover' and step_count < 80:
            state_tensor = env.get_state_tensor()
            action_probs, _ = mcts.search(env)

            states.append(state_tensor)
            policies.append(action_probs)
            current_turns.append(env.current_turn)

            # 特徵探索：前 15 步採用機率採樣，之後選擇最大機率動作
            if step_count < 15:
                valid_mask = env.get_action_mask()
                noise = np.random.dirichlet(0.3 * np.ones(729)) * valid_mask
                probs = 0.75 * action_probs + 0.25 * noise
                sum_p = np.sum(probs)
                if sum_p > 0:
                    probs /= sum_p
                else:
                    probs = valid_mask / np.sum(valid_mask)
                action = np.random.choice(729, p=probs)
            else:
                action = np.argmax(action_probs)

            _, _, done, _ = env.step(action)
            step_count += 1

        # 遊戲結束，結算最終勝分 Target
        scores = env.get_scores()
        total_score = sum(scores) + 1e-5
        outcomes = np.array([s / total_score for s in scores], dtype=np.float32)

        # 保存對局紀錄至 Replay Buffer
        fpath = os.path.join(self.buffer_dir, f'episode_{int(time.time() * 1000)}.json')
        data = {
            'states': [s for s in states],
            'policies': [p.tolist() for p in policies],
            'outcomes': outcomes.tolist(),
            'current_turns': current_turns
        }
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        # 強制清理記憶體垃圾
        del states, policies, current_turns, env, mcts
        gc.collect()
        print(f"[Self-Play] 已完成對局，棋譜已存至: {fpath}")
