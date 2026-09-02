"""AI Chessathon agent.

All chess knowledge comes from model.onnx: a policy+value residual CNN
trained from scratch on high-rated human games (Lichess Elite database).
No external engine is involved at any stage - not in this agent and not
in the training pipeline. Move selection = the network's policy/value
outputs, optionally refined by a small PUCT search (mcts.py) whose
priors and leaf evaluations also come exclusively from the network.
"""
import json
import os
import sys
import time

import chess
import numpy as np
import onnxruntime as ort

import encoding
import mcts

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.onnx")

# The contest runtime provides one dedicated core; keep onnxruntime on it.
_opts = ort.SessionOptions()
_opts.intra_op_num_threads = 1
_opts.inter_op_num_threads = 1
_SESSION = ort.InferenceSession(_MODEL_PATH, _opts, providers=["CPUExecutionProvider"])


def evaluate(boards):
    """Batched network evaluation.

    Returns (priors, values): per board, a dict {legal move -> prior}
    (softmax over legal moves only) and the value in [-1, 1] from the
    mover's perspective. For terminal boards (checkmate/stalemate), returns
    an empty dict {} as priors (no legal moves).
    """
    planes = np.stack([encoding.board_to_planes(b) for b in boards])
    logits, values = _SESSION.run(None, {"planes": planes})
    priors = []
    for board, lg in zip(boards, logits):
        moves = list(board.legal_moves)
        if not moves:
            # Terminal board (checkmate/stalemate) has no legal moves.
            priors.append({})
            continue
        if board.turn == chess.WHITE:
            idx = [encoding.encode_move(m) for m in moves]
        else:
            idx = [encoding.encode_move(encoding.mirror_move(m)) for m in moves]
        raw = lg[idx].astype(np.float64)
        raw = np.exp(raw - raw.max())
        raw /= raw.sum()
        priors.append(dict(zip(moves, raw)))
    return priors, values


def policy_move(board):
    """Single network call; play the highest-prior legal move."""
    priors, _ = evaluate([board])
    return max(priors[0], key=priors[0].get)


_SEARCHER = mcts.Searcher(evaluate, c_puct=1.5, batch_size=16)


def _move_budget(time_left_ms: int) -> float:
    """Seconds to spend on this move. 0.0 = single policy call.

    Conservative by design: flagging loses outright, a slightly shallower
    search only costs a little strength."""
    t = time_left_ms / 1000.0
    if t < 10.0:
        return 0.0
    return min(t / 30.0 + 0.35, t / 4.0, 4.0)


def _choose(board, time_left_ms):
    budget = _move_budget(time_left_ms)
    if budget <= 0.0:
        return policy_move(board)
    return _SEARCHER.search(board, time.perf_counter() + budget)


def get_move(fen: str, time_left_ms: int) -> str:
    """Contest entry point. Never raises; always returns a legal move
    (or 0000 if the position itself is unusable)."""
    try:
        board = chess.Board(fen)
    except Exception:
        return "0000"
    try:
        move = _choose(board, time_left_ms)
        if move in board.legal_moves:
            return move.uci()
    except Exception as exc:  # noqa: BLE001 - losing on error is worse
        print(f"agent error: {exc!r}", file=sys.stderr)
    moves = list(board.legal_moves)
    return moves[0].uci() if moves else "0000"


def main():
    """JSON-lines wire protocol over stdin/stdout, for local simulation."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            move = get_move(req["fen"], int(req["time_left_ms"]))
            print(json.dumps({"move": move}), flush=True)
        except Exception as exc:  # noqa: BLE001 - survive any parsing/format error
            print(json.dumps({"move": "0000"}), flush=True)
            print(f"wire protocol error: {exc!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
