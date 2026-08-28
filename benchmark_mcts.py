import time
import torch
import numpy as np
from framing_env import FramingEnv
from model import FramingAlphaNet
from mcts import MCTS

def benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"運行效能測試工具，計算裝置: {device}")

    model = FramingAlphaNet().to(device)
    model.eval()

    env = FramingEnv()
    mcts = MCTS(model, n_simulations=50, device=device)

    start_t = time.time()
    action_probs, expected_values = mcts.search(env)
    elapsed = (time.time() - start_t) * 1000

    best_action_id = int(np.argmax(action_probs))
    cell_idx = best_action_id // 9
    val = (best_action_id % 9) + 1

    print(f"50 次 MCTS 模擬搜尋完成！耗時: {elapsed:.2f} ms")
    print(f"最佳落子建議: 格子 {cell_idx} (列 {cell_idx//9}, 欄 {cell_idx%9})，數字: {val}")
    print(f"預估三家平分勝率: P0={expected_values[0]:.3f}, P1={expected_values[1]:.3f}, P2={expected_values[2]:.3f}")

if __name__ == '__main__':
    benchmark()
