from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time
from helpers import get_valid_moves, execute_move, random_move
import math
import random

@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super().__init__()
        self.name = "StudentAgent"

    # ---------------- flipping / sandwich bonus ----------------
    @staticmethod
    def flip_bonus(board, move, player):
        opponent = 1 if player == 2 else 2
        r0, c0 = move.get_dest()
        bonus = 0
        directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for dr, dc in directions:
            r, c = r0 + dr, c0 + dc
            count = 0
            while 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                if board[r,c] == opponent:
                    count += 1
                elif board[r,c] == player:
                    bonus += count * 10
                    break
                else:
                    break
                r += dr
                c += dc
        return bonus

    # ---------------- gap / rectangle bonus ----------------
    @staticmethod
    def gap_bonus(board, move, player):
        #reward for filling a gap (single empty space) between two of player's pieces
        #computes bonus after the move is applied
        
        r0, c0 = move.get_dest()
        bonus = 0
        n = board.shape[0]
        directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        for dr, dc in directions:
            for gap in range(1, 4):  # size 1-3
                # forward
                r1, c1 = r0 + dr, c0 + dc
                r2, c2 = r0 + dr*(gap+1), c0 + dc*(gap+1)
                if 0 <= r1 < n and 0 <= c1 < n and 0 <= r2 < n and 0 <= c2 < n:
                    if board[r1,c1] == 0 and board[r2,c2] == player:
                        bonus += 40  # high weight to dominate flips/mobility
                        break
                # backward
                r1n, c1n = r0 - dr, c0 - dc
                r2n, c2n = r0 - dr*(gap+1), c0 - dc*(gap+1)
                if 0 <= r1n < n and 0 <= c1n < n and 0 <= r2n < n and 0 <= c2n < n:
                    if board[r1n,c1n] == 0 and board[r2n,c2n] == player:
                        bonus += 40
                        break
        return bonus

    # ---------------- MCTS Node ----------------
    class Node:
        def __init__(self, board, player, move=None, parent=None):
            self.board = board
            self.player = player
            self.move = move
            self.parent = parent
            self.children = []
            self.untried_moves = get_valid_moves(board, player)
            self.wins = 0.0
            self.visits = 0

        def is_fully_expanded(self):
            return len(self.untried_moves) == 0

        def best_child(self, c=1.3):
            eps = 1e-9
            total_log = math.log(self.visits + 1.0)
            def ucb(child):
                exploit = child.wins / (child.visits + eps)
                explore = c * math.sqrt(2.0 * total_log / (child.visits + eps))
                return exploit + explore
            return max(self.children, key=lambda ch: (ucb(ch), ch.visits))

        def expand(self):
            if not self.untried_moves:
                return None
            move = self.untried_moves.pop(0)
            new_board = deepcopy(self.board)
            execute_move(new_board, move, self.player)
            next_player = 1 if self.player == 2 else 2
            child = StudentAgent.Node(new_board, next_player, move, self)
            self.children.append(child)
            return child

        def backpropagate(self, result):
            self.visits += 1
            self.wins += result
            if self.parent:
                self.parent.backpropagate(result)

    # ---------------- simulation / rollout ----------------
    def simulate(self, board, player, root_player, depth_limit=20):
        corners = {(0,0),(0,board.shape[1]-1),(board.shape[0]-1,0),(board.shape[0]-1,board.shape[1]-1)}
        current_player = player
        depth = 0

        while depth < depth_limit:
            moves = get_valid_moves(board, current_player)
            if not moves:
                current_player = 1 if current_player == 2 else 2
                depth += 1
                continue

            # --- proritize moves that fill gaps ---
            gap_moves = []
            for mv in moves:
                tmp = deepcopy(board)
                execute_move(tmp, mv, current_player)
                if StudentAgent.gap_bonus(tmp, mv, current_player) > 0:
                    gap_moves.append(mv)
            sample_moves = gap_moves if gap_moves else moves

            best_score = -1e9
            best_moves = []

            for mv in sample_moves:
                tmp = deepcopy(board)
                execute_move(tmp, mv, current_player)
                r, c = mv.get_dest()
                score = 0
                #gap priority
                score += StudentAgent.gap_bonus(tmp, mv, current_player)
                #flip / sandwich
                score += StudentAgent.flip_bonus(tmp, mv, current_player) * 0.8
                #corner expansion
                score += 15 if (r,c) in corners else 0
                #mobility
                score += len(get_valid_moves(tmp, current_player)) * 0.2

                if score > best_score:
                    best_score = score
                    best_moves = [mv]
                elif score == best_score:
                    best_moves.append(mv)

            chosen = random.choice(best_moves)
            execute_move(board, chosen, current_player)
            current_player = 1 if current_player == 2 else 2
            depth += 1

        # Final evaluation
        p1 = np.count_nonzero(board == 1)
        p2 = np.count_nonzero(board == 2)
        if root_player == 1:
            return 1.0 if p1 > p2 else 0.5 if p1 == p2 else 0.0
        else:
            return 1.0 if p2 > p1 else 0.5 if p2 == p1 else 0.0

    # ---------------- MCTS ----------------
    def mcts(self, root_board, player, time_limit=1.5):
        root = StudentAgent.Node(deepcopy(root_board), player)
        root_player = player
        start = time.time()
        while time.time() - start < time_limit:
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child(c=1.3)
            if node.untried_moves:
                child = node.expand()
                if child:
                    node = child
            result = self.simulate(deepcopy(node.board), node.player, root_player)
            node.backpropagate(result)

        if not root.children:
            return random_move(root_board, player)
        return max(root.children, key=lambda c: c.visits).move

    # ---------------- step ----------------
    def step(self, board, player, opponent):
        start_time = time.time()
        move = self.mcts(board, player, time_limit=1.5)
        print("MCTS agent decided in", round(time.time() - start_time,3), "seconds.")
        return move
