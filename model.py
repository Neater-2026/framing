import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class FramingAlphaNet(nn.Module):
    """
    AlphaZero 風格 Policy-Value 雙頭卷積神經網路
    輸入: (Batch, 16, 9, 9)
    輸出:
      - policy_logits: (Batch, 729) 代表各動作的先驗概率 (Logits)
      - value: (Batch, 3) 代表 3 位玩家的預期勝率 / 得分期望 (Softmax 分佈)
    """
    def __init__(self, in_channels=16, num_res_blocks=4, num_channels=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_channels)
        
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy Head (策略頭 - 729 種動作)
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 9 * 9, 729)

        # Value Head (價值頭 - 3 玩家勝率/評估分)
        self.value_conv = nn.Conv2d(num_channels, 16, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 3)

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = F.softmax(self.value_fc2(v), dim=-1)

        return policy_logits, value
