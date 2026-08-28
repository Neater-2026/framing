import os
import glob
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from framing_env import FramingEnv
from model import FramingAlphaNet
from mcts import MCTS

class ReplayDataset(Dataset):
    def __init__(self, data_dir='data/replay_buffer', max_files=100):
        self.samples = []
        files = sorted(glob.glob(os.path.join(data_dir, '*.json')), key=os.path.getmtime, reverse=True)[:max_files]
        
        for fpath in files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                states = data['states']
                policies = data['policies']
                outcomes = data['outcomes']

                for s, p in zip(states, policies):
                    self.samples.append((
                        torch.tensor(s, dtype=torch.float32),
                        torch.tensor(p, dtype=torch.float32),
                        torch.tensor(outcomes, dtype=torch.float32)
                    ))
            except Exception as e:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class ContinuousTrainer:
    def __init__(self, checkpoint_dir='checkpoints', device='cpu'):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model = FramingAlphaNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.champion_path = os.path.join(self.checkpoint_dir, 'champion.pt')
        self.metadata_path = os.path.join(self.checkpoint_dir, 'metadata.json')

        if os.path.exists(self.champion_path):
            self.model.load_state_dict(torch.load(self.champion_path, map_location=self.device))
            print(f"已成功載入王者模型: {self.champion_path}")
        else:
            self.save_champion(version="1.0.0", elo=1000)

    def save_champion(self, version, elo):
        torch.save(self.model.state_dict(), self.champion_path)
        meta = {
            'version': version,
            'elo': elo,
            'timestamp': time.time(),
            'model_name': 'FramingAlphaZero-v1'
        }
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        print(f"已發佈新版本 Champion 模型 v{version} (Elo: {elo})")

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for states, target_policies, target_values in dataloader:
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)

            self.optimizer.zero_grad()
            policy_logits, value_preds = self.model(states)

            # Policy Loss (Cross Entropy / KL Divergence)
            log_policies = torch.log_softmax(policy_logits, dim=-1)
            policy_loss = -torch.mean(torch.sum(target_policies * log_policies, dim=-1))

            # Value Loss (MSE)
            value_loss = nn.functional.mse_loss(value_preds, target_values)

            loss = policy_loss + 2.0 * value_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def train_step(self, batch_size=32):
        dataset = ReplayDataset(data_dir=os.path.join(os.path.dirname(__file__), '..', 'data', 'replay_buffer'))
        if len(dataset) == 0:
            return 0.0, 0.0, 0.0
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
        self.model.train()
        for states, target_policies, target_values in loader:
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)

            self.optimizer.zero_grad()
            policy_logits, value_preds = self.model(states)

            log_policies = torch.log_softmax(policy_logits, dim=-1)
            policy_loss = -torch.mean(torch.sum(target_policies * log_policies, dim=-1))
            value_loss = nn.functional.mse_loss(value_preds, target_values)

            loss = policy_loss + 2.0 * value_loss
            loss.backward()
            self.optimizer.step()
            return loss.item(), policy_loss.item(), value_loss.item()
        return 0.0, 0.0, 0.0

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ContinuousTrainer(device=device)

    dataset = ReplayDataset()
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        print(f"開始進行 1 輪增量訓練，數據樣本數: {len(dataset)}...")
        loss = trainer.train_epoch(loader)
        print(f"訓練完成！均方損失 Loss: {loss:.4f}")
    else:
        print("Replay Buffer 中暫無數據，請先運行 self_play.py 生成棋譜。")
