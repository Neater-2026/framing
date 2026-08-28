import math
import numpy as np
import torch

class MCTSNode:
    def __init__(self, state_env, parent=None, action_taken=None, prior=0.0):
        self.state_env = state_env # FramingEnv 副本
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = {}
        self.visit_count = 0
        self.value_sum = np.zeros(3, dtype=np.float32) # P0, P1, P2 價值和

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return np.zeros(3, dtype=np.float32)
        return self.value_sum / self.visit_count

class MCTS:
    """
    3 玩家 MCTS (Monte Carlo Tree Search with PUCT formula)
    """
    def __init__(self, model, c_puct=1.4, n_simulations=100, device='cpu'):
        self.model = model
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.device = device

    @torch.no_grad()
    def search(self, env):
        self.model.eval()
        root = MCTSNode(env)

        # 展開根節點
        self._expand_node(root)

        for _ in range(self.n_simulations):
            node = root
            search_path = [node]

            # 1. Selection (選擇)
            while node.is_expanded() and not node.state_env.game_phase == 'gameover':
                action, node = self._select_child(node)
                search_path.append(node)

            # 2. Expansion & Evaluation (展開與評估)
            value = np.zeros(3, dtype=np.float32)
            if node.state_env.game_phase == 'gameover':
                scores = node.state_env.get_scores()
                total = sum(scores) + 1e-5
                value = np.array([s / total for s in scores], dtype=np.float32)
            else:
                value = self._expand_node(node)

            # 3. Backpropagation (反向傳播)
            for path_node in search_path:
                path_node.visit_count += 1
                path_node.value_sum += value

        # 計算根節點動作選擇分佈
        action_probs = np.zeros(729, dtype=np.float32)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        total_visits = np.sum(action_probs)
        if total_visits > 0:
            action_probs /= total_visits
        else:
            # Fallback uniform over valid actions
            valid_mask = env.get_action_mask()
            action_probs = valid_mask / (np.sum(valid_mask) + 1e-5)

        return action_probs, root.value()

    def _select_child(self, node):
        curr_player = node.state_env.current_turn
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            q_value = child.value()[curr_player]
            u_value = self.c_puct * child.prior * (math.sqrt(node.visit_count) / (1 + child.visit_count))
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand_node(self, node):
        env = node.state_env
        state_tensor = torch.from_numpy(env.get_state_tensor()).unsqueeze(0).to(self.device)
        
        logits, value_pred = self.model(state_tensor)
        logits = logits.squeeze(0).cpu().numpy()
        value = value_pred.squeeze(0).cpu().numpy()

        # 套用 valid action mask
        valid_mask = env.get_action_mask()
        exp_logits = np.exp(logits - np.max(logits)) * valid_mask
        sum_exp = np.sum(exp_logits)
        if sum_exp > 0:
            priors = exp_logits / sum_exp
        else:
            priors = valid_mask / (np.sum(valid_mask) + 1e-5)

        valid_actions = env.get_valid_actions()
        for action in valid_actions:
            next_env = copy_env(env)
            next_env.step(action)
            node.children[action] = MCTSNode(next_env, parent=node, action_taken=action, prior=priors[action])

        return value

def copy_env(env):
    import copy
    new_env = env.__class__()
    new_env.board = [copy.deepcopy(c) for c in env.board]
    new_env.players = [copy.deepcopy(p) for p in env.players]
    new_env.current_turn = env.current_turn
    new_env.game_phase = env.game_phase
    new_env.move_history = list(env.move_history)
    return new_env
