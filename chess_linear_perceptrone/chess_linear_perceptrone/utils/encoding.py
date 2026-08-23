import numpy as np
import chess

# 말 종류별 인덱스 매핑 (순서: 백/흑 폰, 나이트, 비숍, 룩, 퀸, 킹)
PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

# 8x8x12 one-hot 인코딩
# 0~5: 백, 6~11: 흑

def board_to_onehot(board: chess.Board) -> np.ndarray:
    onehot = np.zeros((8, 8, 12), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            offset = 0 if piece.color == chess.WHITE else 6
            idx = PIECE_TO_INDEX[piece.piece_type] + offset
            onehot[row, col, idx] = 1.0
    return onehot.flatten()  # (768,) 