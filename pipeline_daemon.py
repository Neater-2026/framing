import os
import sys
import time
import json
import threading
import copy
import random
import gc
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import FramingAlphaNet
from self_play import SelfPlayWorker
from trainer import ContinuousTrainer
from framing_env import FramingEnv
from mcts import MCTS

class FramingPipelineDaemon:
    """
    純後台 AlphaZero 不間斷深度學習與訓練系統 (記憶體絕不累積版)
    """
    def __init__(self, data_dir='data/replay_buffer', checkpoint_dir='checkpoints', n_simulations=30):
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.n_simulations = n_simulations
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.trainer = ContinuousTrainer(checkpoint_dir=self.checkpoint_dir)
        self.worker = SelfPlayWorker(
            model=self.trainer.model,
            buffer_dir=self.data_dir,
            n_simulations=self.n_simulations
        )

        self.running = False
        self.stats = {
            'episodes_completed': 0,
            'training_steps': 0,
            'current_elo': 1000,
            'model_version': '1.0.0',
            'last_update_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def run_self_play_loop(self):
        """背景自我對弈線程"""
        print("[Daemon] 自我對弈 (Self-Play) 服務已啟動...")
        while self.running:
            try:
                # 同步當前 Champion 模型
                self.worker.model = self.trainer.model
                self.worker.run_episode()
                self.stats['episodes_completed'] += 1
                if self.stats['episodes_completed'] % 5 == 0:
                    print(f"[Self-Play] 已完成 {self.stats['episodes_completed']} 局自自我對弈。")
                gc.collect()
                time.sleep(3) # 釋放 CPU / RAM
            except Exception as e:
                print(f"[Self-Play Error] {e}")
                time.sleep(5)

    def run_training_loop(self):
        """背景訓練與模型迭代線程"""
        print("[Daemon] 模型微調 (Trainer) 服務已啟動...")
        while self.running:
            try:
                # 每隔 20 秒執行一次 Mini-Batch 梯度優化
                time.sleep(20)
                buffer_files = os.listdir(self.data_dir)
                if len(buffer_files) >= 2:
                    loss, p_loss, v_loss = self.trainer.train_step(batch_size=16)
                    self.stats['training_steps'] += 1
                    self.stats['last_update_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[Trainer Step {self.stats['training_steps']}] Loss: {loss:.4f} (Policy: {p_loss:.4f}, Value: {v_loss:.4f})")
                    gc.collect()
            except Exception as e:
                print(f"[Trainer Error] {e}")
                time.sleep(5)

    def start(self):
        self.running = True
        self.t_selfplay = threading.Thread(target=self.run_self_play_loop, daemon=True)
        self.t_train = threading.Thread(target=self.run_training_loop, daemon=True)
        self.t_selfplay.start()
        self.t_train.start()
        print("=== 框制 AlphaZero 後台不間斷學習系統已全面運行 ===")

    def stop(self):
        self.running = False
        print("[Daemon] 後台學習服務已停止。")

if __name__ == '__main__':
    daemon = FramingPipelineDaemon()
    daemon.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        daemon.stop()
