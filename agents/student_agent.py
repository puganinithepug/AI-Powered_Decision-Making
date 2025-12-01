# Alpha beta agent with iterative deepening and a transposition table using zobrist hashing
#
#
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
        self.max_time = 1.9
        
        # transposition table flags are 'EXACT', 'LOWERBOUND', 'UPPERBOUND'
        self.tt = {}
        
        # lazy creation for table
        self.zobrist = None
        self.zobrist_shape = None
        self.random = random.Random(None)

    def zobrist_init(self, board):
        n = board.shape[0]
        if self.zobrist is not None and self.zobrist_shape == (n, n):
            return
        
        self.zobrist_shape = (n, n)
        # cells marked 0, 1, 2 based on occupancy
        self.zobrist = np.zeros((n, n, 3), dtype = np.uint64)
        for i in range(n):
            for j in range(n):
                for p in range(3):
                    self.zobrist[i, j, p] = self.random.getrandbits(64)

    def hash_board(self, board):
        # assigning 0 for empty, 1 and 2 according to player
        n = board.shape[0]
        h = np.uint64(0)

        for i in range(n):
            for j in range(n):
                p = int(board[i, j])
                if p < 0 or p > 2:
                    p = 0
                h ^= np.uint64(self.zobrist[i, j, p])
        return int(h)

    # evaluate heuristics
    def evaluate_board(self, board, color, opponent):
        player_count = int(np.count_nonzero(board == color))
        
        # piece difference
        opp_count = int(np.count_nonzero(board == opponent))
        score_diff = player_count - opp_count
        
        # corner bonus
        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        corner_bonus = sum(1 for (i, j) in corners if board[i, j] == color) * 5
        opp_moves = len(get_valid_moves(board, opponent))
        mobility_penalty = -opp_moves
        
        # bonus for mobility
        my_moves = len(get_valid_moves(board, color))
        mobility_bonus = my_moves * 0.2
        
        return score_diff * 10 + corner_bonus + mobility_penalty + mobility_bonus

    # move ordering helper
    def order_moves(self, board, moves, color, opponent, tt_move = None):
        scored = []
        for m in moves:
            if tt_move is not None and m == tt_move:
                # best possible ordering
                scored.append((9999, m))
                continue

            # move ordering
            new_board = board.copy()
            execute_move(new_board, m, color)
            score = int(np.count_nonzero(new_board == color) - np.count_nonzero(new_board == opponent))
            scored.append((score, m))
        scored.sort(key = lambda x: -x[0])
        return [m for _, m in scored]

    # alpha beta + transposition table
    def alphabeta(self, board, depth, alpha, beta, color, opponent, start_time, max_time):
        if time.time() - start_time > max_time:
            raise TimeoutError()

        self.zobrist_init(board)
        key = self.hash_board(board)
        
        # transposition table lookup
        if key in self.tt:
            entry_depth, entry_score, entry_flag, entry_move = self.tt[key]
            if entry_depth >= depth:
                if entry_flag == 'EXACT':
                    return entry_score, entry_move
                elif entry_flag == 'LOWERBOUND' and entry_score > alpha:
                    alpha = entry_score
                elif entry_flag == 'UPPERBOUND' and entry_score < beta:
                    beta = entry_score
                if alpha >= beta:
                    return entry_score, entry_move

        legal_moves = get_valid_moves(board, color)
        if depth == 0 or not legal_moves:
            val = self.evaluate_board(board, color, opponent)
            return val, None

        best_move = None
        value = -math.inf

        # try transposition table best move ordering
        entry = self.tt.get(key)
        tt_best = entry[3] if entry and entry[3] is not None else None
        moves = self.order_moves(board, legal_moves, color, opponent, tt_best)

        for move in moves:
            if time.time() - start_time > max_time:
                raise TimeoutError()

            new_board = board.copy()
            execute_move(new_board, move, color)

            score_child, _ = self.alphabeta(new_board, depth - 1, -beta, -alpha, opponent, color, start_time, max_time)
            score_child = -score_child

            if score_child > value:
                value = score_child
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        # move storing in tansposition table
        flag = 'EXACT'
        if value <= alpha:
            flag = 'UPPERBOUND'
        elif value >= beta:
            flag = 'LOWERBOUND'
        self.tt[key] = (depth, value, flag, best_move)
        return value, best_move

    # step function
    def step(self, board, color, opponent):
        start_time = time.time()
        max_time = self.max_time

        # initializes zobrist table for our board size
        self.zobrist_init(board)

        legal_moves = get_valid_moves(board, color)
        if not legal_moves:
            return None

        # iterative deepening
        best_move = None
        best_score = -math.inf
        depth = 1
        try:
            while True:
                if depth > 8:
                    break
                val, move = self.alphabeta(board, depth, -math.inf, math.inf, color, opponent, start_time, max_time)
                # only update best_move if the search is completed
                if move is not None:
                    best_move = move
                    best_score = val
                depth += 1
        
        except TimeoutError:
            pass

        # fallback to greedy if a move isnt found in time
        if best_move is None:
            best_score = -math.inf
            for move in legal_moves:
                new_board = board.copy()
                execute_move(new_board, move, color)
                score = self.evaluate_board(new_board, color, opponent)
                if score > best_score:
                    best_score = score
                    best_move = move

            # ensures "None" is never returned
            if best_move is None:
                best_move = legal_moves[0]

        return best_move