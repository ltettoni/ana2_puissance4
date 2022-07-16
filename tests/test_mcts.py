import numpy as np
import pytest
from agents.games_utils import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, GameState


def test__init__():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import initialize_game_state
    blank_board = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    blank_board.fill(NO_PLAYER)
    test_node = Node(initialize_game_state(), PLAYER1)
    assert ((test_node.board == blank_board).all())
    assert test_node.player == PLAYER1
    assert test_node.parent_node is None
    assert test_node.parent_action is None
    assert test_node.children == list()
    assert test_node.nb_visits == 0
    assert test_node.win_count == 0
    assert test_node.loss_count == 0
    assert test_node.exploration_param == np.sqrt(2)
    assert (test_node.untried_actions == [0, 1, 2, 3, 4, 5, 6]).all()
    assert test_node.tried_actions == []
    assert test_node.game_state == GameState.STILL_PLAYING


def test_update_actions():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import initialize_game_state
    test_node = Node(initialize_game_state(), PLAYER1)
    assert test_node.update_actions() == [0, 1, 2, 3, 4, 5, 6]
    assert test_node.tried_actions == []
    test_node.tried_actions = [1, 3]
    assert test_node.update_actions() == [0, 2, 4, 5, 6]


def test_next_player():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import initialize_game_state
    test_node_player1 = Node(initialize_game_state(), PLAYER1)
    test_node_player2 = Node(initialize_game_state(), PLAYER2)
    assert test_node_player1.player == PLAYER1
    assert test_node_player1.next_player() == PLAYER2
    assert test_node_player2.player == PLAYER2
    assert test_node_player2.next_player() == PLAYER1


def test_get_random_untried_action():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import initialize_game_state
    test_node = Node(initialize_game_state(), PLAYER1)
    random_action_test = test_node.get_random_untried_action()
    assert random_action_test < 7
    assert random_action_test >= 0
    assert random_action_test in test_node.untried_actions


def test_is_last_node():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import initialize_game_state
    new_board_full = initialize_game_state()
    new_board_full[0:6, 0:7] = PLAYER1
    full_node = Node(new_board_full, PLAYER2)
    assert full_node.is_last_node() is True
    new_board_won = initialize_game_state()
    new_board_won[0, 0:4] = PLAYER2
    won_node = Node(new_board_won, PLAYER2)
    assert won_node.is_last_node() is True
    new_board_empty = Node(initialize_game_state(), PLAYER1)
    new_board_empty.children.append(won_node)
    assert new_board_empty.is_last_node() is False


def test_final_points():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import initialize_game_state, string_to_board
    board_won_p2 = initialize_game_state()
    board_won_p2[0, 0:4] = PLAYER2
    won_node = Node(board_won_p2, PLAYER2)
    assert won_node.final_points() == 1
    lost_node = Node(board_won_p2, PLAYER1)
    assert lost_node.final_points() == -1
    board_tie_pp = "|==============|\n" \
                   "|O X O O X O X |\n" \
                   "|X O X O O X O |\n" \
                   "|O O O X O X X |\n" \
                   "|X X O X O X O |\n" \
                   "|X O X X X O X |\n" \
                   "|X X X O X X O |\n" \
                   "|==============|\n" \
                   "|0 1 2 3 4 5 6 |\n"
    tie_node = Node(string_to_board(board_tie_pp), PLAYER1)
    assert tie_node.final_points() == -1000
    empty_node = Node(initialize_game_state(), PLAYER1)
    assert empty_node.final_points() == 0


def test_apply_move():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import string_to_board, initialize_game_state
    test_node_empty = Node(initialize_game_state(), PLAYER1)

    new_board = test_node_empty.apply_move(0)
    board_after_move = "|==============|\n" \
                       "|              |\n" \
                       "|              |\n" \
                       "|              |\n" \
                       "|              |\n" \
                       "|              |\n" \
                       "|X             |\n" \
                       "|==============|\n" \
                       "|0 1 2 3 4 5 6 |\n"
    test_board = string_to_board(board_after_move)
    assert ((new_board == test_board).all())
    test_board[0][6] = PLAYER1
    new_node = Node(new_board, PLAYER1)
    test_board_2 = new_node.apply_move(6)
    assert ((test_board_2 == test_board).all())
    with pytest.raises(ValueError):
        test_node_empty.apply_move(8)


def test_tried_all_actions():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import string_to_board, initialize_game_state
    empty_node = Node(initialize_game_state(), PLAYER1)
    assert empty_node.tried_all_actions() is False
    board_pp = "|==============|\n" \
               "|  X X O X O O |\n" \
               "|O O X O O X O |\n" \
               "|O O O X O X X |\n" \
               "|O X O X O X O |\n" \
               "|X O X X X O X |\n" \
               "|X X X O X X O |\n" \
               "|==============|\n" \
               "|0 1 2 3 4 5 6 |\n"
    test_node = Node(string_to_board(board_pp), PLAYER1)
    assert test_node.tried_all_actions() is False
    test_node.board[5][0] = PLAYER1
    assert test_node.tried_all_actions() is True


def test_ucb1():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import initialize_game_state
    from numpy import sqrt, log
    test_fake_parent_node = Node(initialize_game_state(), PLAYER2)
    test_artificial_node = Node(initialize_game_state(), PLAYER1, test_fake_parent_node)
    test_artificial_node.nb_visits = 3
    test_artificial_node.win_count = 2
    test_fake_parent_node.nb_visits = 5
    assert test_artificial_node.ucb1() == 2.0 / 3.0 + sqrt(2) * sqrt(log(5) / 3.0)


def test_best_child():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import initialize_game_state
    test_node = Node(initialize_game_state(), PLAYER1)
    test_node_parent_all = Node(initialize_game_state(), PLAYER2)
    test_node_child1 = Node(initialize_game_state(), PLAYER1, test_node_parent_all)
    test_node_child2 = Node(initialize_game_state(), PLAYER2, test_node_parent_all)
    test_node_child3 = Node(initialize_game_state(), PLAYER1, test_node_parent_all)
    test_node.children = [test_node_child1, test_node_child2, test_node_child3]
    test_node_parent_all.nb_visits = 5
    test_node_child1.nb_visits = 3
    test_node_child1.win_count = 2
    test_node_child2.nb_visits = 10
    test_node_child2.win_count = 1
    test_node_child3.nb_visits = 3
    test_node_child3.win_count = 3
    # child 3 will have the best score since all its visits result in a win
    assert test_node.best_child() == test_node_child3


def test_expand():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import string_to_board

    base_board_pp = "|==============|\n" \
                    "|  X O O X O X |\n" \
                    "|  O X O O X O |\n" \
                    "|  O O X O X X |\n" \
                    "|  X O X O X O |\n" \
                    "|  O X X X O X |\n" \
                    "|  X X O X X O |\n" \
                    "|==============|\n" \
                    "|0 1 2 3 4 5 6 |\n"
    base_board = string_to_board(base_board_pp)

    base_node = Node(base_board, PLAYER1)
    assert base_node.untried_actions == [0]
    assert base_node.tried_actions == []
    child = base_node.expand()

    assert base_node.untried_actions == []
    assert base_node.tried_actions == [0]
    new_board_pp = "|==============|\n" \
                   "|  X O O X O X |\n" \
                   "|  O X O O X O |\n" \
                   "|  O O X O X X |\n" \
                   "|  X O X O X O |\n" \
                   "|  O X X X O X |\n" \
                   "|X X X O X X O |\n" \
                   "|==============|\n" \
                   "|0 1 2 3 4 5 6 |\n"
    new_board = string_to_board(new_board_pp)

    assert base_node.children == [child]
    assert ((child.board == new_board).all())
    assert child.player == PLAYER2
    assert child.parent_node == base_node
    assert child.parent_action == 0
    assert child.children == list()
    assert child.nb_visits == 0
    assert child.win_count == 0
    assert child.loss_count == 0
    assert child.exploration_param == np.sqrt(2)
    assert child.untried_actions == [0]
    assert child.game_state == GameState.STILL_PLAYING


def test_simulation():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import string_to_board
    board_pp = "|==============|\n" \
               "|  X O O X O X |\n" \
               "|  O X O O X O |\n" \
               "|  O O X O X X |\n" \
               "|X X O X O X O |\n" \
               "|X O X X X O X |\n" \
               "|X X X O X X O |\n" \
               "|==============|\n" \
               "|0 1 2 3 4 5 6 |\n"
    test_node_1 = Node(string_to_board(board_pp), PLAYER1)
    assert test_node_1.simulation() == 1

    board_pp_2 = "|==============|\n" \
                 "|              |\n" \
                 "|  O X O O X O |\n" \
                 "|  O O X O O O |\n" \
                 "|  O O X O X O |\n" \
                 "|  X O X X O X |\n" \
                 "|  X X O X X O |\n" \
                 "|==============|\n" \
                 "|0 1 2 3 4 5 6 |\n"

    test_node_2 = Node(string_to_board(board_pp_2), PLAYER2)
    # should end up winning once column 1, 4 or 6 is selected. PLAYER1 should not win. might end in a tie
    result_simulation_2 = test_node_2.simulation()
    assert result_simulation_2 == 1 or result_simulation_2 == -1000

    board_pp_3 = "|==============|\n" \
                 "|    O     O   |\n" \
                 "|  O X O O X O |\n" \
                 "|O O O X O X X |\n" \
                 "|O X O X O X O |\n" \
                 "|O O X X X O X |\n" \
                 "|X X X O X X O |\n" \
                 "|==============|\n" \
                 "|0 1 2 3 4 5 6 |\n"
    test_node_3 = Node(string_to_board(board_pp_3), PLAYER1)
    # player 2 might win (loss of player 1) or it could end in a tie
    result_simulation = test_node_3.simulation()
    assert (result_simulation == -1000) or (result_simulation == -1)


def test_backpropagation():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import string_to_board, initialize_game_state

    test_node_trivial = Node(initialize_game_state(), PLAYER1)
    # test as if we won
    test_node_trivial.backpropagation(1)
    # there were no visits before, so now we have 1
    assert test_node_trivial.nb_visits == 1
    # had 0 win before, now 1
    assert test_node_trivial.win_count == 1

    board_pp_parent = "|==============|\n" \
                      "|  X X O X O O |\n" \
                      "|  O X O O X O |\n" \
                      "|O O O X O X X |\n" \
                      "|O X O X O X O |\n" \
                      "|O O X X X O X |\n" \
                      "|X X X O X X O |\n" \
                      "|==============|\n" \
                      "|0 1 2 3 4 5 6 |\n"
    test_node_parent = Node(string_to_board(board_pp_parent), PLAYER2)
    board_pp_child = "|==============|\n" \
                     "|  X X O X O O |\n" \
                     "|O O X O O X O |\n" \
                     "|O O O X O X X |\n" \
                     "|O X O X O X O |\n" \
                     "|O O X X X O X |\n" \
                     "|X X X O X X O |\n" \
                     "|==============|\n" \
                     "|0 1 2 3 4 5 6 |\n"
    test_node_non_trivial = Node(string_to_board(board_pp_child), PLAYER1, parent_node=test_node_parent,
                                 parent_action=np.int8(0))
    test_node_non_trivial.backpropagation(-1)
    assert test_node_non_trivial.nb_visits == 1
    assert test_node_non_trivial.loss_count == 1
    assert test_node_non_trivial.win_count == 0
    assert test_node_parent.nb_visits == 1
    assert test_node_parent.win_count == 1
    assert test_node_parent.loss_count == 0


def test_select_node_for_simulation():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import string_to_board
    board_pp = "|==============|\n" \
               "|  X X O X O O |\n" \
               "|O O X O O X O |\n" \
               "|O O O X O X X |\n" \
               "|O X O X O X O |\n" \
               "|X O X X X O X |\n" \
               "|X X X O X X O |\n" \
               "|==============|\n" \
               "|0 1 2 3 4 5 6 |\n"
    test_node_full = Node(string_to_board(board_pp), PLAYER2)
    # select_node should expand the previous board and return this new board
    board_pp_after_simulation = "|==============|\n" \
                                "|O X X O X O O |\n" \
                                "|O O X O O X O |\n" \
                                "|O O O X O X X |\n" \
                                "|O X O X O X O |\n" \
                                "|X O X X X O X |\n" \
                                "|X X X O X X O |\n" \
                                "|==============|\n" \
                                "|0 1 2 3 4 5 6 |\n"
    final_node = Node(string_to_board(board_pp_after_simulation), PLAYER1)
    node_selected = test_node_full.select_node_for_simulation()
    assert ((node_selected.board == final_node.board).all())
    assert node_selected.player == PLAYER1
    assert node_selected.parent_node == test_node_full
    assert node_selected.parent_action == 0
    assert node_selected.children == []
    assert node_selected.nb_visits == 0
    assert node_selected.win_count == 0
    assert node_selected.loss_count == 0
    assert node_selected.game_state == GameState.IS_LOST


def test_select_best_action():
    from agents.agent_mcts.mcts import Node
    from agents.games_utils import string_to_board
    test_board = "|==============|\n" \
                 "|              |\n" \
                 "|              |\n" \
                 "|              |\n" \
                 "|      O       |\n" \
                 "|      O       |\n" \
                 "|      O       |\n" \
                 "|==============|\n" \
                 "|0 1 2 3 4 5 6 |\n"
    test_node_1 = Node(string_to_board(test_board), PLAYER2)
    assert test_node_1.select_best_action() == 3
    board_pp = "|==============|\n" \
               "|    O O   O   |\n" \
               "|X O X O X X O |\n" \
               "|O O O X O X X |\n" \
               "|O X O X O X O |\n" \
               "|O O X X X O X |\n" \
               "|X X X O X X O |\n" \
               "|==============|\n" \
               "|0 1 2 3 4 5 6 |\n"
    test_node = Node(string_to_board(board_pp), PLAYER2)
    assert test_node.select_best_action() == 4
