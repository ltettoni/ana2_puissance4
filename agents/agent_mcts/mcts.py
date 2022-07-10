from __future__ import annotations

import numpy as np
from numpy import math

import agents.agent_random.random
from agents.games_utils import BoardPiece, PlayerAction, apply_player_action, PLAYER1, PLAYER2, \
    check_end_state, GameState, Optional, SavedState, Tuple, pretty_print_board
from agents.agent_random.random import legal_actions


def get_random_legal_action(board: np.ndarray) -> PlayerAction:
    legal_actions_randoms = agents.agent_random.random.legal_actions(board)
    random_number = np.random.randint(0, len(legal_actions_randoms))
    random_action = legal_actions_randoms[random_number]
    return np.int8(random_action)


class Node:
    """
    A class used to represent a node in the game tree.
    Attributes :
            board : the node's game board
            player : the current player
            parent_node : the parent node which lead to the current one, default is None
            parent_action : the action that lead to the current node, default is None
            children : possible children of this node
            win_count : number of wins this node leads to
            loss_count : number of losses this node leads to
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
        # print("je construis un node")
        self.board: np.ndarray = board
        self.player: BoardPiece = player
        self.parent_node: Node = parent_node
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
        # print("j'update les actions essayées et pas essayées pour ce node")
        possible_actions = legal_actions(self.board)
        self.untried_actions = [action for action in possible_actions if action not in self.tried_actions]
        self.tried_actions = [action.parent_action for action in self.children]
        return self.untried_actions

    def next_player(self) -> np.int8:
        """
        Returns the next (or previous) player.
        """
        # print("je recherche quel est le prochain joueur")
        return PLAYER1 if self.player == PLAYER2 else PLAYER2

    def get_random_untried_action(self) -> PlayerAction:
        """
        Returns a random column to play in, out of the available ones
        """
        # print("je produis une action random pas encore essayée")
        self.update_actions()
        random_number = np.random.randint(0, len(self.untried_actions))
        random_action = self.untried_actions[random_number]
        # print("c'est l'action : " + str(random_action))
        return np.int8(random_action)

    def is_last_node(self) -> bool:
        """
        Returns true if the current node is at the end of the tree.
        Returns false if the current node can still be expanded.
        """
        # print("je me demande si ce node est le dernier de l'arbre : " + pretty_print_board(self.board))
        # update game state
        self.game_state = check_end_state(self.board, self.player)
        last_node = self.game_state != GameState.STILL_PLAYING
        # print("réponse last node : " + str(last_node))
        return last_node

    def final_points(self) -> int:
        """
        Returns the result of the node : -1 if the game is lost to the other player,
        0 if the game is still playing or ends in a draw and 1 if the game is won.
        """
        # print("je compute le résultat de ce node")
        result = 0
        # find out if other player has won
        end_state_other_player = check_end_state(self.board, self.next_player())
        # update game state
        self.game_state = check_end_state(self.board, self.player)
        if end_state_other_player == GameState.IS_WIN:
            # -1 if the other player wins and we lose
            result = -1
        elif self.game_state == GameState.STILL_PLAYING:
            # 0 if draw or game still on
            result = 0
        elif self.game_state == GameState.IS_DRAW:
            result = -1000
        elif self.game_state == GameState.IS_WIN:
            # 1 if the current player wins
            result = 1
        # print("résultat = " + str(result))
        return result

    def apply_move(self, action) -> np.ndarray:
        """
        Applies the requested action to the board. Doesn't modify the node's board after application.
        Returns the new board after modification.
        :param action: column to be played in
        """
        # print("je met une pièce dans la colonne suivante :" + str(action))
        new_board = apply_player_action(self.board, action, self.player)
        return new_board

    def tried_all_actions(self) -> bool:
        """
        Returns true if we tried all actions available.
        Returns false if there are still actions to be tried
        """
        # print("je me demande si j'ai essayé toutes les actions")
        self.update_actions()
        tried_all = (len(self.untried_actions) == 0)
        # print("réponse tt essayé: " + str(tried_all))
        return tried_all

    def ucb1(self) -> float:
        """
        Returns the result of the upper confidence bound 1 formula for this node.
        """
        # print("je compute le score ucb1 de ce node")
        if self.nb_visits == 0 or self.parent_node.nb_visits == 0:
            raise ValueError
        expectation = float(self.win_count) / float(self.nb_visits)
        exploration = self.exploration_param * np.sqrt(np.log(self.parent_node.nb_visits) / self.nb_visits)
        return expectation + exploration

    def best_child(self) -> Node:
        """
        Returns the child that has the best ucb1 score out of all the node's children.
        """
        # print("je cherche mon best enfant")
        children_ucb1 = [child.ucb1() for child in self.children]
        best_child = self.children[np.argmax(children_ucb1)]
        # print("et c'est avec l'action : " + str(best_child.parent_action))
        return best_child

    def expand(self) -> Node:
        """
        Expands the tree by creating a new node. It is randomly created from the current node.
        Returns the new node.
        """
        # print("j'expand ce node : board :" + pretty_print_board(self.board))
        random_action = self.get_random_untried_action()
        self.tried_actions.append(random_action)
        next_board = self.apply_move(random_action)
        child = Node(next_board, self.next_player(), parent_node=self, parent_action=random_action)
        self.children.append(child)
        self.update_actions()
        return child

    def simulation(self) -> int:
        """
        Simulates a random playout from the current node and returns the result if that playout.
        """
        # print("je simule un résultat pour ce node")
        current_state = self.board
        new_node = Node(current_state, self.player)
        while check_end_state(current_state, self.player) == GameState.STILL_PLAYING:
            # random_action = self.get_random_untried_action()
            random_action = get_random_legal_action(current_state)
            current_state = new_node.apply_move(random_action)
            new_node.board = current_state
        final_points = new_node.final_points()
        # print("le résultat de cette simulation : " + str(final_points))
        return final_points

    def backpropagation(self, result) -> None:
        """
        Backpropagation of the result : update of current node and update of parent nodes.
        """
        # print("je backpropagate un résultat de : " + str(result))
        self.nb_visits += 1
        if result == -1:
            self.loss_count += 1
        elif result == 1:
            self.win_count += 1
        if self.parent_node is not None:
            self.parent_node.backpropagation(result)
        # print("j'ai backpropagate, mon score perso est : " + str(self.win_count))

    def select_node_for_simulation(self) -> Node:
        """
        Selects a node to do a game simulation on.
        Returns the chosen node. If the current node is at the end of the tree, we choose this one.
        """
        # print("je select une node a simuler")
        current_node = self
        while not current_node.is_last_node():
            if not current_node.tried_all_actions():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        # print("board de la node selectionnée : " + pretty_print_board(current_node.board))
        return current_node

    def select_best_action(self) -> PlayerAction:
        """
        Selects the best action for the current board.
        Returns the most optimal action to play.
        """
        # print("je cherche la meilleure action")
        simulation_nb = 100
        for i in range(simulation_nb):
            new_node = self.select_node_for_simulation()
            result = new_node.simulation()
            new_node.backpropagation(result)
        best_child = self.best_child()
        action = best_child.parent_action
        print("meilleure action : " + str(action))
        print("arbre de merde qui m'a permis de prendre cette décision : ")
        print("ceci est l'arbre : " + pretty_print_board(self.board))
        for i in self.children:
            print("étage 1 des enfants :")
            print(pretty_print_board(i.board))
            for j in i.children:
                print("étage 2 des enfants :")
                print(pretty_print_board(j.board))
                for k in j.children:
                    print("étage 3 des enfants :")
                    print(pretty_print_board(k.board))
                    for a in k.children:
                        print("étage 4 des enfants :")
                        print(pretty_print_board(a.board))
        return action


def generate_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> \
        Tuple[PlayerAction, Optional[SavedState]]:
    root = Node(board, player)
    action = root.select_best_action()
    return np.int8(action), saved_state
