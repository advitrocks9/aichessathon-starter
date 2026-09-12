"""The submission entrypoint. The platform imports this file and calls get_move."""

import random

import chess

# Import time runs before the clock starts. 90s in the qualifier, 30s at the final.


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in, and your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive, but suspended between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game

    print() is safe. Your stdout is redirected away from the protocol stream and kept in a
    log only your team can read, after validation and after every rated game.
    """
    board = chess.Board(fen)

    # TODO: Replace everything below this and build a winning bot!
    return random.choice(list(board.legal_moves)).uci()
