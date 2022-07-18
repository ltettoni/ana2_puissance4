from enum import Enum
import numpy as np

from typing import Callable, Tuple, Optional

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

NB_ROWS = 6  # number of rows in board
NB_COLS = 7  # number of columns in board
TOP_BOARD = NB_ROWS - 1  # row at the top of the board

NB_ROWS_PP = 9  # number of rows in printed board (borders included)
NB_COLS_PP = 17  # number of columns in printed board (borders, spaces and line break included)

CONNECT_N = 4  # number of pieces required to win


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """

    game_state = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    game_state.fill(NO_PLAYER)
    return game_state


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """

    # fill an array with spaces
    pp_board = [NO_PLAYER_PRINT] * (NB_COLS_PP * NB_ROWS_PP)
    # adding top frame
    pp_board[:NB_COLS_PP + 1] = "|==============|\n"
    # adding middle and borders
    # row from 1 to 6 (rows), col from 0 to 16 (cols)
    for row in range(1, NB_ROWS_PP - 2):
        for col in range(NB_COLS_PP):
            current_position = get_position(row, col)
            # put borders and line break
            if col == 0 or col == 15:
                pp_board[current_position] = '|'
            elif col == 16:
                pp_board[current_position] = '\n'
            # symbols only on odd cols, leave a space on even cols
            elif col % 2 == 1:
                index = int(col / 2)
                # get element in the given board
                elem = board[6 - row][index]
                if elem == PLAYER1:
                    pp_board[current_position] = PLAYER1_PRINT
                elif elem == PLAYER2:
                    pp_board[current_position] = PLAYER2_PRINT

    # adding bottom of frame
    pp_board[-(2 * NB_COLS_PP) + 1:] = "|==============|\n|0 1 2 3 4 5 6 |\n"
    return "".join(pp_board)


def get_position(row: int, col: int) -> int:
    """
    Returns the position in array used to create the pretty print board, given the indexes
    """

    return NB_COLS_PP * row + col


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """

    board = initialize_game_state()
    # row from 1 to 6 (rows), col from 1 to 16 (cols)
    for row in range(1, NB_ROWS_PP - 2):
        for col in range(1, NB_COLS_PP):
            # elements only on odd columns
            if col % 2 == 1:
                elem = pp_board[get_position(row, col)]
                index = int(col / 2)
                if elem == PLAYER1_PRINT:
                    board[6 - row][index] = PLAYER1
                elif elem == PLAYER2_PRINT:
                    board[6 - row][index] = PLAYER2
                elif elem == NO_PLAYER_PRINT:
                    board[6 - row][index] = NO_PLAYER
    return board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. Raises a ValueError
    if action is not a legal move. If it is a legal move, the modified version of the
    board is returned and the original board should remain unchanged (i.e., either set
    back or copied beforehand).
    """

    if action >= NB_COLS or board[TOP_BOARD][action] != NO_PLAYER:
        raise ValueError

    open_row = min(np.argwhere(board[:, action] == NO_PLAYER).flatten())

    final_board = board.copy()
    final_board[open_row][action] = player
    return final_board


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """

    if check_lines(board, player):
        return True
    if check_columns(board, player):
        return True
    if check_diagonals(board, player):  # ascending diagonals
        return True
    if check_diagonals(np.fliplr(board), player):  # descending diagonals
        return True
    return False


def check_columns(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Checks if the lines of the given board contain a winning connected line of 4 pieces.
    """

    for row in range(NB_ROWS - (CONNECT_N - 1)):
        for col in range(NB_COLS):
            consecutive = True
            for incr in range(CONNECT_N):
                if board[row + incr][col] != player:
                    consecutive = False
                    break
            if consecutive: return True
    return False


def check_lines(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Checks if the lines of the given board contain a winning connected line of 4 pieces.
    """

    for row in range(NB_ROWS):
        for col in range(NB_COLS - (CONNECT_N - 1)):
            consecutive = True
            for incr in range(CONNECT_N):
                if board[row][col+incr] != player:
                    consecutive = False
                    break
            if consecutive: return True
    return False


def check_diagonals(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Checks if the diagonals of the given board contain a winning connected diagonal of 4 pieces.
    """
    for row in range(NB_ROWS - (CONNECT_N - 1)):
        for col in range(NB_COLS - (CONNECT_N - 1)):
            consecutive = True
            for incr in range(CONNECT_N):
                if board[row+incr][col+incr] != player:
                    consecutive = False
                    break
            if consecutive: return True
    return False


def check_board_full(board: np.ndarray) -> bool:
    """
    Checks if the given board is full and has no space left for new moves
    """
    return (board != NO_PLAYER).all()


def other_player(curr_player: BoardPiece) -> BoardPiece:
    return PLAYER1 if curr_player == PLAYER2 else PLAYER2


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the specified player, i.e. has their last
    action in the specified board won (GameState.IS_WIN) or drawn (GameState.IS_DRAW),
    or is play still on-going (GameState.STILL_PLAYING)?
    """

    # Start checking with a potential more frequent case
    if connected_four(board, player):
        return GameState.IS_WIN
    # Then a full board situation, that happens less frequently
    if check_board_full(board):
        return GameState.IS_DRAW
    # Otherwise, game is open and we are still playing
    return GameState.STILL_PLAYING
