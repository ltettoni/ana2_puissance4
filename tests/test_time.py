from typing import Optional

disable_jit = True
if disable_jit:
    import os

    os.environ['NUMBA_DISABLE_JIT'] = '1'

import timeit
import numpy as np
from numba import njit
from scipy.signal.sigtools import _convolve2d
from agents.games_utils import connected_four, initialize_game_state, BoardPiece, PlayerAction, NO_PLAYER, CONNECT_N, PLAYER1


@njit()
def connected_four_iter(
        board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None
) -> bool:
    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1

    for i in range(rows):
        for j in range(cols_edge):
            if np.all(board[i, j: j + CONNECT_N] == player):
                return True

    for i in range(rows_edge):
        for j in range(cols):
            if np.all(board[i: i + CONNECT_N, j] == player):
                return True

    for i in range(rows_edge):
        for j in range(cols_edge):
            block = board[i: i + CONNECT_N, j: j + CONNECT_N]
            if np.all(np.diag(block) == player):
                return True
            if np.all(np.diag(block[::-1, :]) == player):
                return True

    return False


col_kernel = np.ones((CONNECT_N, 1), dtype=BoardPiece)
row_kernel = np.ones((1, CONNECT_N), dtype=BoardPiece)
dia_l_kernel = np.diag(np.ones(CONNECT_N, dtype=BoardPiece))
dia_r_kernel = np.array(np.diag(np.ones(CONNECT_N, dtype=BoardPiece))[::-1, :])


def connected_four_convolve(
        board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None
) -> bool:
    board = board.copy()

    other_player = BoardPiece(player % 2 + 1)
    board[board == other_player] = NO_PLAYER
    board[board == player] = BoardPiece(1)

    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board, kernel, 1, 0, 0, BoardPiece(0))
        if np.any(result == CONNECT_N):
            return True
    return False


def test_time():
    board = initialize_game_state()
    board[0, 0] = PLAYER1
    board[1, 1] = PLAYER1
    board[2, 2] = PLAYER1
    board[3, 3] = PLAYER1
    number = 10 ** 4

    res = timeit.timeit("connected_four_iter(board, player)",
                        setup="connected_four_iter(board, player)",
                        number=number,
                        globals=dict(connected_four_iter=connected_four_iter,
                                     board=board,
                                     player=BoardPiece(1)))

    print(f"Python iteration-based: {res / number * 1e6 : .1f} us per call")

    res = timeit.timeit("connected_four_convolve(board, player)",
                        number=number,
                        globals=dict(connected_four_convolve=connected_four_convolve,
                                     board=board,
                                     player=BoardPiece(1)))

    print(f"Convolve2d-based: {res / number * 1e6 : .1f} us per call")

    res = timeit.timeit("connected_four(board, player)",
                        setup="connected_four(board, player)",
                        number=number,
                        globals=dict(connected_four=connected_four,
                                     board=board,
                                     player=BoardPiece(1)))

    print(f"My secret sauce: {res / number * 1e6 : .1f} us per call")
