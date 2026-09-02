import numpy as np
from board import (
    board_from_fen,
    board_to_fen,
    print_board,
    piece_on,
    get_side,
    get_ep_square,
    get_castling,
    CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ
)

# Standard Chess FENs for testing edge cases
TEST_FENS = [
    # 1. Starting Position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # 2. Midgame with active En Passant square (e3)
    "r1bqk2r/pp1ppp1p/2n2np1/8/3NP3/2P1B3/P1P2PPP/R2QKB1R w KQkq e3 0 8",
    # 3. Partial Castling Rights (only White Kingside & Black Queenside)
    "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",
    # 4. No Castling Rights, Black to move
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b - - 0 1"
]


def test_fen_roundtrip():
    """Verify that parsing a FEN and converting it back yields the exact same string."""
    print("--- 1. Testing FEN Round-trip Parsing ---")
    all_passed = True
    for i, original_fen in enumerate(TEST_FENS, 1):
        board = board_from_fen(original_fen)
        reconstructed_fen = board_to_fen(board)
        
        if original_fen == reconstructed_fen:
            print(f" Test {i}: PASSED")
        else:
            print(f" Test {i}: FAILED")
            print(f"   Original:      {original_fen}")
            print(f"   Reconstructed: {reconstructed_fen}")
            all_passed = False
    
    assert all_passed, "Some FEN round-trip tests failed!"


def test_board_state_assertions():
    """Verify specific bitboard fields are stored correctly."""
    print("\n--- 2. Testing Internal Board State ---")
    
    # Test Starting Position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = board_from_fen(start_fen)

    # 1. Side to move
    assert get_side(board) == 0, "Expected White to move (0)"
    
    # 2. En Passant sentinel
    assert get_ep_square(board) == 64, "Expected EP to be 64 (none)"

    # 3. Castling rights (should be 15 or 0b1111)
    castling = get_castling(board)
    assert castling & CASTLE_WK, "Missing White Kingside Castling"
    assert castling & CASTLE_WQ, "Missing White Queenside Castling"
    assert castling & CASTLE_BK, "Missing Black Kingside Castling"
    assert castling & CASTLE_BQ, "Missing Black Queenside Castling"

    # 4. Specific Piece Locations (a1 = 0 should be White Rook = 3)
    assert piece_on(board, 0) == 3, f"Expected White Rook (3) at a1 (0), got {piece_on(board, 0)}"
    # e8 = 60 should be Black King = 11
    assert piece_on(board, 60) == 11, f"Expected Black King (11) at e8 (60), got {piece_on(board, 60)}"

    print(" All internal state assertions PASSED!")


def test_visual_print():
    """Print the starting position to manually inspect ascii art output."""
    print("\n--- 3. Visual Board Output Test ---")
    board = board_from_fen(TEST_FENS[0])
    print_board(board)


if __name__ == "__main__":
    test_fen_roundtrip()
    test_board_state_assertions()
    test_visual_print()
    print("\nAll board unit tests passed successfully!")