import random
from typing import Union

import chess
import torch
from chess import Board

import numpy as np
from torch import nn

import ChessPositionEvaluator


def sigmoid_inv(y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    denom = np.maximum(1-y, np.full(y.shape, fill_value=0.00001))
    return np.log(y/denom)


def piece_evaluate(board: Board) -> float:
    net_evaluation = 0

    # piece evaluation
    piece_values = [-1, -3, -3, -5, -9, -1000, 1, 3, 3, 5, 9, 1000]
    piece_value_tensor = np.array([np.full(shape=(8, 8), fill_value=value) for value in piece_values])
    flat_board_tensor = get_bitboards(board).reshape(-1)
    flat_piece_value_tensor = piece_value_tensor.reshape(-1)
    piece_evaluation = float(np.dot(flat_board_tensor, flat_piece_value_tensor))
    net_evaluation += piece_evaluation

    if board.is_game_over():
        if board.outcome().result() == '1-0':
            net_evaluation = 1000000
        elif board.outcome().result() == '0-1':
            net_evaluation = -1000000

    return net_evaluation


def learned_evaluate(board: Board, model: nn.Module) -> float:
    # model = ChessPositionEvaluator.ResNetEvaluator(device='cuda:0')
    # model.load_state_dict(torch.load(r'.\last_net.pth'))
    bitboard_tensor = torch.tensor(get_bitboards(board, with_turn=True), dtype=torch.float32)
    bitboard_tensor = torch.reshape(bitboard_tensor, (-1, 13, 8, 8))
    yhat = model(bitboard_tensor).detach().numpy()
    yhat = sigmoid_inv(yhat)
    # noise = 0.0005
    noise = 0
    yhat = yhat[0][0]
    return yhat + random.random() * noise - noise/2


def multiple_learned_evaluate(boards: list[Board], model: nn.Module) -> np.ndarray:
    bitboard_tensors = []
    if len(boards) == 0:
        print('uh oh')
    for board in boards:
        bitboard_tensor = get_bitboards(board, with_turn=True)
        bitboard_tensors.append(bitboard_tensor)
    bitboard_tensors = torch.tensor(np.array(bitboard_tensors), dtype=torch.float32)
    # print('shape:', bitboard_tensors.shape)
    yhat = model(bitboard_tensors).detach().numpy()
    yhat = sigmoid_inv(yhat)
    yhat = yhat.reshape(-1)
    # print('shape np:', yhat.shape)
    return yhat


def get_bitboards(board: Board, with_turn: bool = False) -> np.ndarray:
    # get bitboards from Board
    black, white = board.occupied_co
    bitboards = np.array([
        black & board.pawns,
        black & board.knights,
        black & board.bishops,
        black & board.rooks,
        black & board.queens,
        black & board.kings,
        white & board.pawns,
        white & board.knights,
        white & board.bishops,
        white & board.rooks,
        white & board.queens,
        white & board.kings,
    ], dtype=np.uint64)

    # convert bitboards to tensor
    bitboards = np.asarray(bitboards, dtype=np.uint64)
    bitboards = bitboards.reshape(-1, 1)
    shift = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    bitboards = (bitboards >> shift).astype(np.uint8)
    bitboards = np.unpackbits(bitboards, bitorder='little')
    bitboards = bitboards.reshape((-1, 8, 8))

    if with_turn:
        # rotate board if black
        if board.turn is chess.BLACK:
            bitboards = np.flip(bitboards, axis=1)
            bitboards = np.flip(bitboards, axis=2)
        # add turn and piece evaluation info
        # turn_bitboard = np.zeros(shape=(1, 8, 8))
        # piece_evaluation = piece_evaluate(board)
        turn_value = 1 if board.turn is chess.WHITE else -1
        # turn_bitboard[0, 0, 0] = turn_value
        # turn_bitboard[0, 0, 1] = piece_evaluation
        turn_bitboard = np.full(shape=(1, 8, 8), fill_value=turn_value)
        bitboards = np.concatenate((bitboards, turn_bitboard))

    return bitboards


def get_child_boards(current_board: Board) -> list[Board]:
    legal_moves = list(iter(current_board.legal_moves))
    child_boards = [current_board.copy(stack=False) for _ in range(len(legal_moves))]
    [board.push(move) for (board, move) in zip(child_boards, legal_moves)]
    return child_boards
