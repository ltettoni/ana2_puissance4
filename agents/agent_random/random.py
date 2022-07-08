import numpy as np
import random

from agents.games_utils import PlayerAction, BoardPiece, Optional, SavedState, Tuple, NO_PLAYER


def legal_actions(board) -> list[np.int8]:
    """
    Returns an array of the non-full columns where a move can be played.
    """
    return np.argwhere(board[-1, :] == NO_PLAYER).flatten()


def generate_move_random(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    """
    Generates a random move in one of the available columns.
    """
    valid_action = legal_actions(board)
    action = random.choice(valid_action)
    return action, saved_state
