import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MultilayerPerceptronAgent(nn.Module):
    def __init__(self, state_dim=768, hidden_dim=256, lr=1e-3, gamma=0.99, device='cpu'):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim, bias=True)  # 첫 번째 층
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)          # 두 번째 층
        self.gamma = gamma
        self.device = device
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        # state: (batch, 768) or (768,)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        x = torch.relu(self.fc1(state))  # ReLU 활성화 함수
        return self.fc2(x).squeeze(-1)   # (batch,)

    def act(self, state, legal_moves, env, epsilon=0.1):
        # epsilon-greedy 정책
        if np.random.rand() < epsilon:
            return np.random.choice(legal_moves)
        q_values = []
        for move in legal_moves:
            # move 적용 후 상태 예측
            board_copy = env.board.copy()
            board_copy.push_uci(move)
            next_state = env.__class__()
            next_state.board = board_copy
            state_vec = next_state.reset()  # one-hot
            q = self.forward(state_vec).item()
            q_values.append(q)
        max_idx = int(np.argmax(q_values))
        return legal_moves[max_idx]

    def update(self, state, action, reward, next_state, done):
        # Q-learning 업데이트
        self.optimizer.zero_grad()
        q_pred = self.forward(state)
        with torch.no_grad():
            q_next = self.forward(next_state)
            q_target = reward if done else reward + self.gamma * q_next.item()
        loss = nn.functional.mse_loss(q_pred, torch.tensor([q_target], device=self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item() 