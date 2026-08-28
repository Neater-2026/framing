import numpy as np

class FramingEnv:
    """
    Framing 9x9 棋盤遊戲 Python 環境模組
    狀態空間: 9x9 (81 格) 盤面，3 位玩家 (0, 1, 2)，每人初始持有 1~9 各 3 張牌
    動作空間: 81 x 9 = 729 種離散動作 (action_id = cell_index * 9 + (value - 1))
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [None] * 81
        self.players = [
            {'score': 0, 'hand': [3] * 9},
            {'score': 0, 'hand': [3] * 9},
            {'score': 0, 'hand': [3] * 9}
        ]
        self.current_turn = 0
        self.game_phase = 'playing'
        self.move_history = []
        return self.get_state_tensor()

    def is_perfect_square(self, num: int) -> bool:
        if num <= 0:
            return False
        sqrt = int(math.isqrt(num))
        return sqrt * sqrt == num

    def find_valid_frames(self, current_board, target_index):
        import math
        row = target_index // 9
        col = target_index % 9
        valid_frames = []

        for size in range(2, 10):
            min_r = max(0, row - size + 1)
            max_r = min(9 - size, row)
            min_c = max(0, col - size + 1)
            max_c = min(9 - size, col)

            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    sum_val = 0
                    is_full = True
                    color_counts = {0: 0, 1: 0, 2: 0}
                    cells = []

                    for i in range(size):
                        for j in range(size):
                            cell_idx = (r + i) * 9 + (c + j)
                            cell = current_board[cell_idx]
                            if cell is None:
                                is_full = False
                                break
                            sum_val += cell['value']
                            color_counts[cell['player']] += 1
                            cells.append(cell_idx)
                        if not is_full:
                            break

                    if is_full:
                        sqrt_val = int(math.isqrt(sum_val))
                        if sqrt_val * sqrt_val == sum_val and sum_val > 0:
                            max_cards = -1
                            winners = []
                            for pid in range(3):
                                if color_counts[pid] > max_cards:
                                    max_cards = color_counts[pid]
                                    winners = [pid]
                                elif color_counts[pid] == max_cards:
                                    winners.append(pid)
                            
                            points = sum_val // len(winners)
                            valid_frames.append({
                                'size': size,
                                'sum': sum_val,
                                'color_counts': color_counts,
                                'cells': cells,
                                'winners': winners,
                                'points_per_winner': points
                            })
        return valid_frames

    def get_valid_actions(self, player_idx=None):
        if self.game_phase == 'gameover':
            return []

        if player_idx is None:
            player_idx = self.current_turn

        hand = self.players[player_idx]['hand']
        avail_vals = [val for val in range(1, 10) if hand[val - 1] > 0]
        if not avail_vals:
            return []

        valid_actions = []
        for cell_idx in range(81):
            if self.board[cell_idx] is None:
                for val in avail_vals:
                    action_id = cell_idx * 9 + (val - 1)
                    valid_actions.append(action_id)
        return valid_actions

    def get_action_mask(self, player_idx=None):
        mask = np.zeros(729, dtype=np.float32)
        valid_actions = self.get_valid_actions(player_idx)
        for act in valid_actions:
            mask[act] = 1.0
        return mask

    def step(self, action_id: int):
        if self.game_phase == 'gameover':
            return self.get_state_tensor(), 0, True, {}

        cell_index = action_id // 9
        value = (action_id % 9) + 1

        player = self.players[self.current_turn]
        if player['hand'][value - 1] <= 0 or self.board[cell_index] is not None:
            # 無效落子懲罰
            return self.get_state_tensor(), -10.0, True, {'error': 'Invalid action'}

        player['hand'][value - 1] -= 1
        self.board[cell_index] = {'player': self.current_turn, 'value': value}
        self.move_history.append((self.current_turn, cell_index, value))

        # 檢測與結算框制
        frames = self.find_valid_frames(self.board, cell_index)
        points_earned = [0, 0, 0]
        if frames:
            for f in frames:
                for w_id in f['winners']:
                    self.players[w_id]['score'] += f['points_per_winner']
                    points_earned[w_id] += f['points_per_winner']

        prev_turn = self.current_turn
        self.next_turn()
        done = (self.game_phase == 'gameover')

        reward = float(points_earned[prev_turn])
        return self.get_state_tensor(), reward, done, {'frames': frames, 'points': points_earned}

    def next_turn(self):
        attempts = 0
        next_p = (self.current_turn + 1) % 3
        while attempts < 3:
            has_cards = any(c > 0 for c in self.players[next_p]['hand'])
            has_empty = any(b is None for b in self.board)
            if has_cards and has_empty:
                self.current_turn = next_p
                return
            next_p = (next_p + 1) % 3
            attempts += 1
        self.game_phase = 'gameover'

    def get_state_tensor(self) -> np.ndarray:
        """
        轉換盤面為 16 通道 9x9 特徵矩陣:
        [0..2]: P0, P1, P2 佔據格子的數字 (normalized / 9.0)
        [3..5]: P0, P1, P2 佔據狀態 mask (1.0 代表有棋子)
        [6..8]: P0 剩餘手牌數量特徵 (1-3, 4-6, 7-9)
        [9..11]: P1 剩餘手牌數量特徵
        [12..14]: P2 剩餘手牌數量特徵
        [15]: 當前回合玩家指示器 (P0: 0.0, P1: 0.5, P2: 1.0)
        """
        tensor = np.zeros((16, 9, 9), dtype=np.float32)

        # 盤面特徵
        for idx in range(81):
            r, c = idx // 9, idx % 9
            cell = self.board[idx]
            if cell is not None:
                pid = cell['player']
                val = cell['value']
                tensor[pid, r, c] = val / 9.0
                tensor[pid + 3, r, c] = 1.0

        # 手牌特徵 (分 3 組填滿全圖)
        for pid in range(3):
            hand = self.players[pid]['hand']
            base_ch = 6 + pid * 3
            tensor[base_ch, :, :] = np.mean(hand[0:3]) / 3.0
            tensor[base_ch + 1, :, :] = np.mean(hand[3:6]) / 3.0
            tensor[base_ch + 2, :, :] = np.mean(hand[6:9]) / 3.0

        # 當前輪次指示
        tensor[15, :, :] = self.current_turn / 2.0

        return tensor
