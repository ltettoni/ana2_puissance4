from __future__ import annotations

import numpy as np
from numpy import math

from agents.games_utils import BoardPiece, PlayerAction, apply_player_action, PLAYER1, PLAYER2, \
    check_end_state, GameState, Optional, SavedState, Tuple, pretty_print_board, initialize_game_state
from agents.agent_random.random import legal_actions


class Node:
    """
    A class used to represent a node in the game tree.
    Attributes :
            board : the node's game board
            player : the current player (will play the next round)
            parent_node : the parent node which lead to the current one, default is None
            level : depth in the tree - starts with 1 - to display the tree clearly
            parent_action : the action that lead to the current node, default is None
            children : possible children of this node
            nb_visits : number of times this node has been visited
            win_count : number of wins this node leads to, according to the node's player
            loss_count : number of losses this node leads to, according to the node's player
            exploration_param : by default sqrt(2)
            untried_actions : a list of columns which can be played and haven't been expanded yet
            tried_actions : a list of columns where the node has been expanded already
            game_state : current state of the game
    """

    def __init__(self, board: np.ndarray, player: BoardPiece, parent_node: Node = None,
                 parent_action: PlayerAction = None) -> None:
        """
        Constructor function for the class Node
        """

        self.board: np.ndarray = board
        self.player: BoardPiece = player
        self.parent_node: Node = parent_node
        self.level: int = 1 if parent_node is None else parent_node.level + 1
        self.parent_action: BoardPiece = parent_action
        self.children: list[Node] = []
        self.nb_visits: int = 0
        self.win_count: int = 0
        self.loss_count: int = 0
        self.exploration_param: float = math.sqrt(2)
        self.untried_actions: list[PlayerAction] = legal_actions(self.board)
        self.tried_actions: list[PlayerAction] = []
        self.game_state: GameState = check_end_state(board, self.player)
        return

    def update_actions(self) -> list[PlayerAction]:
        """
        Updates the tried and untried actions of the current node. To be used after we expanded the node or before
        getting a random action.
        Returns the list of untried actions.
        """
        possible_actions = legal_actions(self.board)
        self.tried_actions = [child.parent_action for child in self.children]
        self.untried_actions = [action for action in possible_actions if action not in self.tried_actions]
        return self.untried_actions

    def next_player(self) -> np.int8:
        """
        Returns the next (or previous) player.
        """

        return PLAYER1 if self.player == PLAYER2 else PLAYER2

    def get_random_untried_action(self) -> PlayerAction:
        """
        Returns a random column to play in, out of the available ones.
        Available if a child hasn't been created from this action yet.
        """

        self.update_actions()
        random_number = np.random.randint(0, len(self.untried_actions))
        random_action = self.untried_actions[random_number]
        return np.int8(random_action)

    def game_is_over(self) -> bool:
        """
        Updates the game state and returns true if the game is over.
        Returns false if the game is still playing.
        """

        self.game_state = check_end_state(self.board, self.player)
        other_player_game_state = check_end_state(self.board, self.next_player())
        game_over = (self.game_state != GameState.STILL_PLAYING) or (other_player_game_state != GameState.STILL_PLAYING)
        return game_over

    def final_points(self) -> int:
        """
        Returns the result of the board of the node:
         -1 if the game is lost to the other player,
         0 if the game is still playing or ends in a draw,
         +1 if the game is won.
        """
        # find out if other player has won
        end_state_other_player = check_end_state(self.board, self.next_player())
        # update game state for us current player
        self.game_state = check_end_state(self.board, self.player)

        result = 0
        if end_state_other_player == GameState.IS_WIN:
            # -1 if the game is lost to the other player
            result = -1
        elif self.game_state == GameState.STILL_PLAYING or self.game_state == GameState.IS_DRAW:
            # 0 if the game is still playing or ends in a draw
            result = 0
        elif self.game_state == GameState.IS_WIN:
            # +1 if the game is won
            result = 1
        return result

    def apply_move(self, action) -> np.ndarray:
        """
        Applies the requested action to the board. Doesn't modify the node's board after application.
        Returns the new board after modification.
        :param action: column to be played in
        """

        new_board = apply_player_action(self.board, action, self.player)
        return new_board

    def tried_all_actions(self) -> bool:
        """
        Updates the tried and untried actions.
        Returns true if we tried all actions available.
        Returns false if there are still actions to be tried.
        """

        self.update_actions()
        return len(self.untried_actions) == 0

    def ucb1(self) -> float:
        """
        Returns the result of the upper confidence bound 1 formula for this node, using the win_count.
        """

        # raises a ValueError to avoid a division by zero
        if self.nb_visits == 0 or self.parent_node.nb_visits == 0:
            raise ValueError
        expectation = float(self.win_count) / float(self.nb_visits)
        exploration = self.exploration_param * np.sqrt(np.log(self.parent_node.nb_visits) / self.nb_visits)
        return expectation + exploration

    def ucb1_alternative(self) -> float:
        """
        Returns the result of the upper confidence bound 1 formula for this node, using lose_count instead of win_count.
        The idea is that the lose_count of this node corresponds to the wins for the parent node (because they depend
        on different players).
        """

        # raises a ValueError to avoid a division by zero
        if self.nb_visits == 0 or self.parent_node.nb_visits == 0:
            raise ValueError
        expectation = float(self.loss_count) / float(self.nb_visits)
        exploration = self.exploration_param * np.sqrt(np.log(self.parent_node.nb_visits) / self.nb_visits)
        return expectation + exploration

    def best_child(self) -> Node:
        """
        Returns the child that has the best ucb1 score out of all the node's children.
        """

        children_ucb1 = [child.ucb1() for child in self.children]
        best_child = self.children[np.argmax(children_ucb1)]
        return best_child

    def best_child_alternative(self) -> Node:
        """
        Returns the child that has the best ucb1 alternative score out of all the node's children.
        """

        children_ucb1_alt = [child.ucb1_alternative() for child in self.children]
        best_child_alt = self.children[np.argmax(children_ucb1_alt)]
        return best_child_alt

    def select_node_for_simulation(self) -> Node:
        """
        Selects a node to do a game simulation (step 1 of MCTS).
        Wikipedia reads:
        "select successive child nodes until a leaf node L is reached. The root is the current game state and a leaf is
        any node that has a potential child from which no simulation (playout) has yet been initiated."
        Returns the chosen node. If the current node is at the end of the tree, we choose this one.
        """

        running_node = self
        # to go down in tree : the node has to have expanded all its children (not be a "leaf") and not be a terminal state
        while running_node.tried_all_actions() and not running_node.game_is_over():
            # go down by choosing the best child
            running_node = running_node.best_child()
        if running_node.game_is_over():
            return running_node
        else:
            return running_node.expand()

    def expand(self) -> Node:
        """
        Expands the tree by creating a new node (step 2 of MCTS).
        It is randomly created from the current node by choosing a random untried action.
        Returns the new node.
        """

        random_action = self.get_random_untried_action()
        next_board = self.apply_move(random_action)
        child = Node(next_board, self.next_player(), parent_node=self, parent_action=random_action)
        self.children.append(child)
        self.update_actions()
        return child

    def simulation(self) -> int:
        """
        Simulates a random playout from the current node, until the game is over (step 3 of MCTS).
        Returns the result of that playout (-1, 0 or 1), from the current player's point of view.
        """

        current_board = self.board
        # initialize the running node with the current node's information (start playing with tue current player)
        running_node = Node(current_board, self.player)

        # Simulate each player in turn, until the current_board can no longer be played
        while check_end_state(current_board, running_node.player) == GameState.STILL_PLAYING:
            random_action = get_random_legal_action(current_board)
            current_board = running_node.apply_move(random_action)
            # Update the running node to continue the simulation with next player
            running_node.board = current_board
            running_node.player = running_node.next_player()
        # Now the game can no longer be played, we compute the score from the initial player's point of view
        running_node.player = self.player
        final_points = running_node.final_points()
        return final_points

    def backpropagation(self, result) -> None:
        """
        Backpropagation of the result : update of current node and then all its
        parent nodes up to the root (step 4 of MCTS).
        """

        self.nb_visits += 1
        if result == -1:
            self.loss_count += 1
        elif result == 1:
            self.win_count += 1

        # Recursive backpropagation to the parent
        if self.parent_node is not None: # we are not the root
            # backpropagate the opposite result to the parent, because it has the other player as its player.
            self.parent_node.backpropagation(-result)

    def select_best_action(self) -> PlayerAction:
        """
        Selects the best action for the current board. Puts all the step of MCTS together and does them for a
        fixed number of iterations.
        Returns the most optimal action to play.
        """

        # Maybe be here instead of fixing the number of loops we could measure the current system time
        # and iterate while the simulation did not exceed a given period, say, 1 second of real-time.

        simulation_nb = 1500
        for simulation in range(simulation_nb):
            new_node = self.select_node_for_simulation()
            result = new_node.simulation()
            new_node.backpropagation(result)
        best_child = self.best_child_alternative()
        action = best_child.parent_action
        # for debugging / understanding purposes
        # self.display_full_tree()
        return action

    # UTILITY METHODS TO DISPLAY THE TREE

    def display_node(self):
        """
        Displays a node and its information. To be used in display_tree
        """
        print("level=" + str(self.level) + ", action=" + str(
            self.parent_action) + ", actionsToHere=" + self.actions_to_here() + " score=" + str(self.win_count) + "/" +
              str(self.nb_visits) + ", nextPlayer=" + str(self.player))
        print(pretty_print_board(self.board))

    def actions_to_here(self):
        """
        Shows the successive actions which lead to the board.
        """
        acc: str = ""
        running = self
        while running is not None:
            acc = str(running.parent_action) + acc
            running = running.parent_node
        return acc.removeprefix("None")

    # Recursive depth-first is not ideal to represent the tree
    def display_tree(self, max_depth: int, depth: int) -> None:
        # displays the tree
        if depth >= max_depth:
            return
        self.display_node()
        for child in self.children:
            child.display_tree(max_depth, depth + 1)

    # Only display nodes at a given level, starting with decounter=1
    def display_tree_at_level(self, decounter: int) -> None:
        if decounter == 1:
            self.display_node()
        for child in self.children:
            child.display_tree_at_level(decounter - 1)

    # Breadth first display (root node, then all level 2, then all level 3, etc)
    def display_full_tree(self) -> None:
        print("\n------------------------ FULL TREE ------------------------")
        for level in range(1, 10):
            self.display_tree_at_level(level)


def get_random_legal_action(board: np.ndarray) -> PlayerAction:
    """
    Returns a random legal action for this board. To be used in the simulation step.
    """
    curr_legal_actions = legal_actions(board)
    random_number = np.random.randint(0, len(curr_legal_actions))
    random_action = curr_legal_actions[random_number]
    return np.int8(random_action)


def generate_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> \
        Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generates the best move for the current board.
    """
    # force 1st move in column 3 because it's the best move possible
    if (board == initialize_game_state()).all():
        action = 3
    else:
        root = Node(board, player)
        action = root.select_best_action()
    return np.int8(action), saved_state
