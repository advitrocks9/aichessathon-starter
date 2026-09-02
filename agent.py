"""The submission entrypoint. The platform imports this file and calls get_move."""

import random

from board import (
    board_from_fen,
    move_to_uci,
    # generate_legal_moves,  <-- Import your move generator function when built
)

# =============================================================================
# WARMUP / PRE-COMPUTATION (Runs once on import within the 60s setup budget)
# =============================================================================
# Trigger Numba's JIT compilation here so your bot doesn't lose time 
# compiling functions during the first move's clock.
_dummy_board = board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

# Call your JIT search / movegen once to force compilation ahead of time
# _ = generate_legal_moves(_dummy_board)


# Import time runs once per game, inside a 60 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.
    
    fen: the current position
    time_left_ms: remaining clock time in milliseconds
    returns: "e2e4", "g1f3", "e7e8q", etc.
    """
    # 1. Parse the incoming FEN string into your Numba board array
    board = board_from_fen(fen)

    # 2. Generate legal moves for the current position
    # legal_moves = generate_legal_moves(board)

    # -------------------------------------------------------------------------
    # 3. Search / Decision Logic (Placeholder)
    # Pass 'board' and 'time_left_ms' into your minimax/alphabeta or NN evaluation.
    # chosen_move = search(board, legal_moves, time_left_ms)
    # -------------------------------------------------------------------------
    
    # Example placeholder return until movegen is plugged in:
    # return move_to_uci(chosen_move)
    pass
