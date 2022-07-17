import numpy as np
import pytest
from agents.games_utils import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, GameState


def test_initialize_game_state():
    from agents.games_utils import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board():
    from agents.games_utils import pretty_print_board
    from agents.games_utils import initialize_game_state

    init_board = initialize_game_state()
    init_pp = pretty_print_board(init_board)
    assert init_pp == "|==============|\n|              |\n" \
                      "|              |\n|              |\n" \
                      "|              |\n|              |\n" \
                      "|              |\n|==============|" \
                      "\n|0 1 2 3 4 5 6 |\n"
    init_board[0, 1] = PLAYER2
    init_board[2, 4] = PLAYER1
    example_pp = pretty_print_board(init_board)
    assert example_pp == "|==============|\n" \
                         "|              |\n" \
                         "|              |\n" \
                         "|              |\n" \
                         "|        X     |\n" \
                         "|              |\n" \
                         "|  O           |\n" \
                         "|==============|\n" \
                         "|0 1 2 3 4 5 6 |\n"

    init_board[5, 5] = PLAYER1
    ex_pp = pretty_print_board(init_board)
    assert ex_pp == "|==============|\n" \
                    "|          X   |\n" \
                    "|              |\n" \
                    "|              |\n" \
                    "|        X     |\n" \
                    "|              |\n" \
                    "|  O           |\n" \
                    "|==============|\n" \
                    "|0 1 2 3 4 5 6 |\n"

    init_board.fill(PLAYER1)
    full_pp = pretty_print_board(init_board)
    assert full_pp == "|==============|\n" \
                      "|X X X X X X X |\n" \
                      "|X X X X X X X |\n" \
                      "|X X X X X X X |\n" \
                      "|X X X X X X X |\n" \
                      "|X X X X X X X |\n" \
                      "|X X X X X X X |\n" \
                      "|==============|\n" \
                      "|0 1 2 3 4 5 6 |\n"


def test_string_to_board():
    from agents.games_utils import initialize_game_state
    from agents.games_utils import string_to_board

    ret = initialize_game_state()
    str_init = "|==============|\n|              |\n" \
               "|              |\n|              |\n" \
               "|              |\n|              |\n" \
               "|              |\n|==============|\n" \
               "|0 1 2 3 4 5 6 |\n"
    assert np.all(string_to_board(str_init) == ret)

    ret[0, 1] = PLAYER2
    ret[2, 4] = PLAYER1
    ret[5, 5] = PLAYER1
    str_test = "|==============|\n" \
               "|          X   |\n" \
               "|              |\n" \
               "|              |\n" \
               "|        X     |\n" \
               "|              |\n" \
               "|  O           |\n" \
               "|==============|\n" \
               "|0 1 2 3 4 5 6 |\n"
    assert np.all(string_to_board(str_test) == ret)
    ret.fill(PLAYER2)
    str_full = "|==============|\n" \
               "|O O O O O O O |\n" \
               "|O O O O O O O |\n" \
               "|O O O O O O O |\n" \
               "|O O O O O O O |\n" \
               "|O O O O O O O |\n" \
               "|O O O O O O O |\n" \
               "|==============|\n" \
               "|0 1 2 3 4 5 6 |\n"
    assert np.all(string_to_board(str_full) == ret)


def test_apply_player_action():
    from agents.games_utils import apply_player_action
    from agents.games_utils import initialize_game_state
    ret = initialize_game_state()

    board1 = apply_player_action(ret, np.int8(0), PLAYER1)
    # check ret unchanged
    assert np.all(ret == initialize_game_state())
    ret[0][0] = PLAYER1
    assert np.all(board1 == ret)
    board2 = apply_player_action(board1, np.int8(0), PLAYER2)
    ret[1][0] = PLAYER2
    assert np.all(board2 == ret)

    ret[2][0] = PLAYER1
    ret[3][0] = PLAYER2
    ret[4][0] = PLAYER1
    ret[5][0] = PLAYER2
    # illegal moves
    with pytest.raises(ValueError):
        apply_player_action(ret, np.int8(0), PLAYER1)
    with pytest.raises(ValueError):
        apply_player_action(ret, np.int8(8), PLAYER1)


def test_connected_four():
    from agents.games_utils import connected_four
    from agents.games_utils import initialize_game_state
    init = initialize_game_state()

    assert not connected_four(init, PLAYER1)
    assert not connected_four(init, PLAYER2)

    # check cols
    init[2: 2 + 4, 5] = PLAYER2
    assert connected_four(init, PLAYER2)
    assert not connected_four(init, PLAYER1)

    # check row
    init[0, 0: 0 + 4] = PLAYER1
    assert connected_four(init, PLAYER1)

    # check diagonals
    init2 = initialize_game_state()
    init2[5][0] = PLAYER1
    init2[4][1] = PLAYER1
    init2[3][2] = PLAYER1
    init2[2][3] = PLAYER1
    assert connected_four(init2, PLAYER1)
    assert not connected_four(init2, PLAYER2)

    init2[2][0] = PLAYER2
    init2[3][1] = PLAYER2
    init2[4][2] = PLAYER2
    init2[5][3] = PLAYER2
    assert connected_four(init2, PLAYER2)


def test_check_end_state():
    from agents.games_utils import check_end_state
    from agents.games_utils import initialize_game_state, string_to_board
    ret = initialize_game_state()

    assert GameState.STILL_PLAYING == check_end_state(ret, PLAYER1)
    assert GameState.STILL_PLAYING == check_end_state(ret, PLAYER2)

    ret[0, 0: 4] = PLAYER1

    assert GameState.IS_WIN == check_end_state(ret, PLAYER1)

    ret[2: 2 + 4, 0] = PLAYER2

    assert GameState.IS_WIN == check_end_state(ret, PLAYER2)

    ret.fill(PLAYER1)
    assert GameState.IS_WIN == check_end_state(ret, PLAYER1)

    bug_board = "|==============|\n" \
                "|      X       |\n" \
                "|      X       |\n" \
                "|      X       |\n" \
                "|    O O       |\n" \
                "|    O X       |\n" \
                "|  O X O X     |\n" \
                "|==============|\n" \
                "|0 1 2 3 4 5 6 |"

    assert GameState.STILL_PLAYING == check_end_state(string_to_board(bug_board), PLAYER1)

def test_check_board_full():
    from agents.games_utils import check_board_full
    from agents.games_utils import initialize_game_state
    board = initialize_game_state()
    assert not check_board_full(board)
    board.fill(PLAYER1)
    assert check_board_full(board)
    board.fill(PLAYER2)
    assert check_board_full(board)
    board.fill(NO_PLAYER)
    assert not check_board_full(board)
