import chess
import numpy as np
from utils.encoding import board_to_onehot

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return board_to_onehot(self.board)

    def step(self, move_uci):
        """
        move_uci: UCI 포맷의 수(ex: 'e2e4')
        반환: (다음 상태 벡터, 보상, 게임 종료 여부, info)
        """
        move = chess.Move.from_uci(move_uci)
        if move not in self.board.legal_moves:
            # 불법 수: 큰 패널티
            return board_to_onehot(self.board), -10.0, True, {'illegal': True}
        self.board.push(move)
        done = self.board.is_game_over()
        reward = self._get_reward(done)
        return board_to_onehot(self.board), reward, done, {'illegal': False}

    def legal_moves(self):
        return [move.uci() for move in self.board.legal_moves]

    def render(self):
        print(self.board)

    def _get_reward(self, done):
        if not done:
            return 0.0
        result = self.board.result()
        if result == '1-0':
            return 1.0  # 백 승
        elif result == '0-1':
            return -1.0  # 흑 승
        else:
            return 0.0  # 무승부 