import os
import json
import time
import numpy as np
import torch
from framing_env import FramingEnv
from model import FramingAlphaNet
from mcts import MCTS

class SelfPlayWorker:
    """
    自我對弈與 Replay Buffer 數據生成器
    """
    def __init__(self, model, buffer_dir='data/replay_buffer', n_simulations=50, device='cpu'):
        self.model = model
        self.buffer_dir = buffer_dir
        self.n_simulations = n_simulations
        self.device = device
        os.makedirs(self.buffer_dir, exist_ok=True)

    def run_episode(self):
        env = FramingEnv()
        mcts = MCTS(self.model, n_simulations=self.n_simulations, device=self.device)

        states = []
        policies = []
        current_turns = []

        step_count = 0
        while not env.game_phase == 'gameover' and step_count < 100:
            state_tensor = env.get_state_tensor()
            action_probs, _ = mcts.search(env)

            states.append(state_tensor)
            policies.append(action_probs)
            current_turns.append(env.current_turn)

            # 特徵探索：前 15 步採用機率採樣，之後選擇最大機率動作
            if step_count < 15:
                # Add Dirichlet noise for exploration
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

        episode_data = {
            'states': [s.tolist() for s in states],
            'policies': [p.tolist() for p in policies],
            'current_turns': current_turns,
            'outcomes': outcomes.tolist(),
            'final_scores': scores
        }

        # 儲存 Replay 數據檔
        filename = os.path.join(self.buffer_dir, f'episode_{int(time.time()*1000)}.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f)

        return len(states), scores

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FramingAlphaNet().to(device)
    worker = SelfPlayWorker(model, n_simulations=20, device=device)
    print("啟動單局 Self-Play 模擬...")
    steps, scores = worker.run_episode()
    print(f"自我對弈完成！共 {steps} 步，最終得分: {scores}")
