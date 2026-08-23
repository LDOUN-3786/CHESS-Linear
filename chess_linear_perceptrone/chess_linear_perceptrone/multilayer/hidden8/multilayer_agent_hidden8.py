import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MultilayerPerceptronAgentHidden8(nn.Module):
    def __init__(self, state_dim=768, hidden_dim=8, num_hidden_layers=8, lr=1e-3, gamma=0.99, device='cpu'):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.gamma = gamma
        self.device = device
        
        # 은닉층 8개로 구성
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim, bias=True))
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
        
        layers.append(nn.Linear(hidden_dim, 1, bias=True))
        
        self.network = nn.Sequential(*layers)
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        # state: (batch, 768) or (768,)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        x = state
        for i, layer in enumerate(self.network):
            if i < len(self.network) - 1:  # 마지막 층 제외하고 ReLU 적용
                x = torch.relu(layer(x))
            else:
                x = layer(x)
        
        return x.squeeze(-1)   # (batch,)

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