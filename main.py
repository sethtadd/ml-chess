import multiprocessing
from multiprocessing import pool
from collections.abc import Callable
from functools import partial
from random import randrange

import chess
import torch
from chess import Board
from chess.engine import SimpleEngine
import numpy as np

import ChessPositionEvaluator
import board_utils
from board_utils import piece_evaluate, get_child_boards


board_scores = {}
boards_to_evaluate = []

model = ChessPositionEvaluator.ResNetEvaluator(device='cuda:0')
model.eval()
model.load_state_dict(torch.load(r'.\last_net.pth'))


def base_minimax(board: chess.Board, depth: int, white_turn: bool, evaluate_board: Callable = piece_evaluate):
    prepopulate_minimax(board=board, depth=depth, white_turn=white_turn)
    if len(boards_to_evaluate) != 0:
        evaluations = board_utils.multiple_learned_evaluate(boards_to_evaluate, model)
        for evaluation, board_hash in zip(evaluations, map(Board.fen, boards_to_evaluate)):
            board_scores[board_hash] = evaluation
        boards_to_evaluate.clear()
    return minimax(board=board, depth=depth, white_turn=white_turn, evaluate_board=evaluate_board)


def prepopulate_minimax(board: chess.Board, depth: int, white_turn: bool):
    if depth == 0 or board.is_game_over():
        board_hash = board.fen()
        if board_hash not in board_scores:
            boards_to_evaluate.append(board)
        return

    if white_turn:
        for child_board in get_child_boards(board):
            prepopulate_minimax(child_board, depth - 1, False)
    else:  # black's turn
        for child_board in get_child_boards(board):
            prepopulate_minimax(child_board, depth - 1, True)


def minimax(board: chess.Board, depth: int, white_turn: bool, evaluate_board: Callable = piece_evaluate) -> float:
    if depth == 0 or board.is_game_over():
        board_hash = board.fen()
        if board_hash not in board_scores:
            print('--------- NOT CACHED???')
            board_scores[board_hash] = evaluate_board(board)
        return board_scores[board_hash]

    if white_turn:
        value = -np.Inf
        for child_board in get_child_boards(board):
            value = max(value, minimax(child_board, depth - 1, False, evaluate_board))
    else:  # black's turn
        value = np.Inf
        for child_board in get_child_boards(board):
            value = min(value, minimax(child_board, depth - 1, True, evaluate_board))

    return value


def minimax_parallel(board: chess.Board, depth: int, white_turn: bool, evaluate_board: Callable = piece_evaluate) -> float:
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    # partial minimax func that just takes board as input
    minimax_partial = partial(minimax, depth=depth - 1, white_turn=white_turn)  # TODO just pass *args
    child_boards = get_child_boards(board)
    evaluations = list(multiprocessing.pool.Pool(processes=len(child_boards)).map(func=minimax_partial, iterable=child_boards))  # FIXME BSOD happens here

    if white_turn:
        return max(evaluations)
    else:  # black's turn
        return min(evaluations)


def get_engine_move(board: chess.Board, engine: chess.engine.SimpleEngine) -> chess.Move:
    return engine.play(board, chess.engine.Limit(depth=0)).move


def get_engines_fish_7_vs_15(board: Board, engine7, engine15) -> chess.Move:
    if board.turn is chess.WHITE:
        return engine7.play(board, chess.engine.Limit(depth=10)).move
    else:
        return engine15.play(board, chess.engine.Limit(depth=10)).move


def get_naive_move(board: Board, evaluator: Callable = piece_evaluate, depth: int = 3) -> chess.Move:
    moves = list(iter(board.legal_moves))

    move_evaluations = []
    for move in moves:
        new_board = board.copy(stack=False)
        new_board.push(move)
        white_turn = new_board.turn is chess.WHITE
        move_evaluations.append(base_minimax(board=new_board, depth=depth, white_turn=white_turn, evaluate_board=evaluator))
        # move_evaluations.append(minimax_parallel(board=new_board, depth=3, white_turn=white_turn))  # FIXME this BSOD's for some reason

    if board.turn is chess.WHITE:
        best_move_idx = move_evaluations.index(max(move_evaluations))
    else:  # black's turn
        best_move_idx = move_evaluations.index(min(move_evaluations))

    return moves[best_move_idx]


def get_move_against_user(board: Board) -> chess.Move:
    learned_evaluate_partial = partial(board_utils.learned_evaluate, model=model)
    if board.turn == chess.WHITE:
        move = get_naive_move(board, depth=2, evaluator=learned_evaluate_partial)
    else:
        try:
            usr_san = input('ur move bitch: ')
            move = board.parse_san(usr_san)
            valid_move = True
        except ValueError:
            valid_move = False
            print('that\'s not a move move dummy')

        while not valid_move:
            try:
                usr_san = input('ur move bitch: ')
                move = board.parse_san(usr_san)
                valid_move = True
            except ValueError:
                valid_move = False
                print('that\'s not a move move dummy')
    return move


def get_random_move(board: chess.Board) -> chess.Move:
    moves = list(iter(board.legal_moves))
    move = moves[randrange(len(moves))]
    return move


def main():

    board = chess.Board()

    # show initial board
    print(board)
    print()

    pgn_moves = []

    engine15 = SimpleEngine.popen_uci(r'C:\Users\setht\Downloads\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe')
    engine7 = SimpleEngine.popen_uci(r'C:\Users\setht\Downloads\stockfish-7-win\stockfish-7-win\Windows\stockfish 7 x64.exe')

    # iterate through turns
    while not board.is_game_over():
        # update board with move
        # move = get_move_against_user(board)
        learned_evaluate_partial = partial(board_utils.learned_evaluate, model=model)
        move = get_naive_move(board, depth=2, evaluator=learned_evaluate_partial)
        # move = get_engines_fish_7_vs_15(board, engine7, engine15)
        # move = get_engine_move(board, engine15)
        # move = get_naive_move(board)
        # move = get_random_move(board)
        pgn_moves.append(board.san(move))
        board.push(move)

        # show turn info
        print(f'half-move: {board.ply()}')
        print(f'evaluation: {board_utils.learned_evaluate(board, model)}')
        print(f'fen: {board.fen()}')
        print(board)
        print()

    # show game over info
    print(f'game over: {str(board.outcome().termination).removeprefix("Termination.")}')
    print(f'outcome: {board.outcome().result()}')

    engine15.quit()
    engine7.quit()

    with open('moves.pgn', 'w') as file:
        file.write('\n'.join(pgn_moves))


if __name__ == '__main__':
    main()
