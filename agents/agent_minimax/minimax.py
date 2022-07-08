import sys
import numpy as np

from agents.agent_random.random import legal_actions
from agents.games_utils import PLAYER1, PLAYER2, BoardPiece, GameState, Optional, SavedState, Tuple, \
    PlayerAction
from agents.games_utils import check_end_state, apply_player_action

DEPTH = 4
ALPHA = -100
BETA = 100
INIT_PREVIOUS_INDEX = -1
STILL_PLAYING_NEGAMAX = 101


def generate_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> \
        Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generates the best possible move for the player, using the negamax function.
    """

    (action, value) = negamax(board, DEPTH, ALPHA, BETA, player, INIT_PREVIOUS_INDEX)
    return np.int8(action), saved_state


def negamax(current_board: np.ndarray, depth: int, alpha: int, beta: int, player: BoardPiece, previous_index: int) -> \
        (int, int):
    """
    Generates the index_col and the value of the best move in the current board.
    """

    other_player = PLAYER1 if player == PLAYER2 else PLAYER2

    end_state = check_end_state(current_board, other_player)
    if depth == 0 or end_state != GameState.STILL_PLAYING:
        return previous_index, heuristic(current_board, player, depth, end_state)

    valid_cols = legal_actions(current_board)
    reordered_cols = reorder_columns(valid_cols)

    value = ~sys.maxsize
    index_col = -1

    for col in reordered_cols:
        new_board = apply_player_action(current_board, col, player)

        value_negamax = -negamax(new_board, depth - 1, -beta, -alpha, other_player, col)[1]
        if value_negamax > value:
            value = value_negamax
            index_col = col

        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return index_col, value


def heuristic(current_board: np.ndarray, player: BoardPiece, depth: int, end_state: GameState):
    """
    Heuristic used for the negamax algorithm.
    """
    other_player = PLAYER1 if player == PLAYER2 else PLAYER2
    end_state_other = check_end_state(current_board, other_player)
    if end_state == GameState.IS_WIN:
        return BETA * (depth + 1)
    elif end_state_other == GameState.IS_WIN:
        return ALPHA * (depth + 1)
    elif end_state == GameState.IS_DRAW:
        return 0
    # game state is still playing but the depth was 0
    else:
        return STILL_PLAYING_NEGAMAX * (depth + 1)


def reorder_columns(cols: list) -> list:
    """
    Reorders the columns so that we start in the middle and go outwards. If there are
    only two columns we start with the 2nd one.
    """
    size = len(cols)
    middle = size // 2
    reordered_cols = []
    for i in range(middle):
        reordered_cols.append(cols[middle - i])
        if size % 2 != 0 or i != middle - 1:
            reordered_cols.append(cols[middle + (i + 1)])

    reordered_cols.append(cols[0])
    return reordered_cols
