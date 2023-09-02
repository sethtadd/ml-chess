from time import time
from typing import Union

import chess
import numpy as np
from chess import Board
from chess.engine import Limit
from chess.engine import SimpleEngine

from board_utils import get_bitboards


CPU_COUNT = 8  # fixme should be 10 but bsod's


def generate_game_board_positions(engine_exe: Union[str, tuple[str, str]], limit: Limit, games: int = 1) -> (set, Board, float):
    # initialize engine(s)
    if type(engine_exe) is str:  # one engine for white and black
        engine_w = engine_b = SimpleEngine.popen_uci(engine_exe)
        engine_w.configure({'Threads': CPU_COUNT})
    else:  # tuple, white and black have different engines
        engine_w = SimpleEngine.popen_uci(engine_exe[0])
        engine_b = SimpleEngine.popen_uci(engine_exe[1])
        engine_w.configure({'Threads': CPU_COUNT})
        engine_b.configure({'Threads': CPU_COUNT})

    # play games
    positions = set()
    board = Board()
    start_time = time()
    while not board.is_game_over():
        engine_current = engine_w if board.turn is chess.WHITE else engine_b  # determine which engine's turn it is
        move = engine_current.play(board, limit).move  # update board
        board.push(move)
        positions.add(board.fen())
    end_time = time()

    duration = end_time - start_time

    # terminate engine(s)
    if engine_b is not engine_w:
        engine_b.quit()
    engine_w.quit()

    return positions, board, duration


def generate_multiple_game_board_positions(games: int = 1, **kwargs):
    all_positions = set()
    for _ in range(games):
        game_positions, _, _ = generate_game_board_positions(**kwargs)
        all_positions = all_positions.union(game_positions)

    return all_positions


def board_tensors_and_scores_from_positions(positions: set[str], engine_exe: str, limit: Limit) -> (np.ndarray, np.ndarray):
    # create ordered list of Boards from fen positions
    positions = [Board(fen=position_fen) for position_fen in positions]
    tensors = np.array([get_bitboards(board, with_turn=True) for board in positions])

    # score each board
    engine = SimpleEngine.popen_uci(engine_exe)
    engine.configure({'Threads': CPU_COUNT})
    scores = [engine.analyse(board, limit)['score'].white() for board in positions]
    engine.quit()
    # convert mates to scores of +/- 1,000,000
    scores = [(1000000 if '#+' in str(score) else score) for score in scores]
    scores = [(-1000000 if '#-' in str(score) else score) for score in scores]
    # convert to ints
    scores = [int(str(score)) for score in scores]
    scores = np.array(scores)

    return tensors, scores


def generate_data_test():
    stockfish7_exe = r'C:\Users\setht\Downloads\stockfish-7-win\stockfish-7-win\Windows\stockfish 7 x64.exe'
    stockfish15_exe = r'C:\Users\setht\Downloads\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe'
    print('simulating games...')
    # positions, final_board, total_time = generate_game_board_positions((stockfish7_exe, stockfish15_exe), Limit(depth=10))
    # positions, final_board, total_time = generate_game_board_positions(stockfish15_exe, Limit(depth=10))
    positions = generate_multiple_game_board_positions(games=100, engine_exe=stockfish15_exe, limit=Limit(time=0.01))

    # print(final_board)
    # print(str(final_board.outcome().termination).removeprefix("Termination."))
    # print(final_board.result())
    #
    # print('moves:', final_board.ply())
    # print('duration:', total_time)
    # print('duration per move:', total_time/final_board.ply())

    print('positions:')
    print(len(positions))

    print('scoring positions...')
    board_tensors, board_scores = board_tensors_and_scores_from_positions(positions, stockfish15_exe, Limit(depth=6))

    print('sorted:')
    print(sorted(board_scores))

    np.save(file='./board_tensors.npy', arr=board_tensors, allow_pickle=False)
    np.save(file='./board_scores.npy', arr=board_scores, allow_pickle=False)


def load_data_test():
    board_tensors = np.load(file='./board_tensors.npy', mmap_mode='r')
    board_scores = np.load(file='./board_scores.npy', mmap_mode='r')

    print('size:', len(board_scores))
    print('dims:', board_tensors.shape)

    # for tensor, score in zip(board_tensors, board_scores):
    #     print('score:', score)
    #     print('tensor:')
    #     print(tensor)


generate_data_test()
# load_data_test()
