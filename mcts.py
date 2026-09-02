"""Small batched PUCT search (the AlphaZero recipe, sized for one CPU core).

Every prior and every leaf evaluation comes from the network via the
evaluate() callable - there is no handcrafted evaluation anywhere.
Leaves are collected in small batches using virtual loss so the ONNX
runtime evaluates several positions per call.
"""
import time

_VIRTUAL_LOSS = 1.0


class _Node:
    __slots__ = ("prior", "visits", "value_sum", "children")

    def __init__(self, prior):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children = None  # dict[chess.Move, _Node] once expanded


class Searcher:
    def __init__(self, evaluate, c_puct=1.5, batch_size=16):
        self.evaluate = evaluate
        self.c_puct = c_puct
        self.batch_size = batch_size

    def search(self, board, deadline):
        root = _Node(1.0)
        priors, _ = self.evaluate([board])
        root.children = {m: _Node(p) for m, p in priors[0].items()}
        if not root.children:
            # Terminal root (checkmate/stalemate): no legal move exists to
            # return. Fail fast instead of spinning batches until deadline.
            raise ValueError("search() called on a position with no legal moves")
        if len(root.children) == 1:
            return next(iter(root.children))

        while True:
            leaves, terminal_backups = self._run_batch(board, root)
            if not leaves and not terminal_backups:
                break  # no progress possible
            if time.perf_counter() >= deadline:
                break
        return max(root.children, key=lambda m: root.children[m].visits)

    def _run_batch(self, board, root):
        """Collect up to batch_size distinct leaves with virtual loss,
        evaluate them in one network call, back up the results.

        Visit accounting: a child's visits are incremented exactly once per
        traversal, at selection time (that increment doubles as the virtual
        loss); backup only settles value_sum. A parent's visit total is the
        sum of its children's visits, computed in _select_child."""
        leaves, terminal_backups = [], 0
        seen = set()
        for _ in range(self.batch_size):
            node, path = root, []
            while node.children:
                node = self._select_child(node, path)
                board.push(path[-1][0])
            outcome = board.outcome()
            stop = False
            if outcome is not None:
                # Terminal: exact value, no net call. Mated mover scores -1.
                self._backup(path, -1.0 if outcome.winner is not None else 0.0)
                terminal_backups += 1
            elif id(node) in seen:
                # Same unexpanded leaf selected twice: abandon this path and
                # evaluate what we have (more selections would repeat it).
                self._undo_virtual(path)
                stop = True
            else:
                seen.add(id(node))
                leaves.append((node, path))
            for _ in range(len(path)):
                board.pop()
            if stop:
                break

        if leaves:
            boards = []
            for _, path in leaves:
                b = board.copy(stack=False)
                for mv, _ in path:
                    b.push(mv)
                boards.append(b)
            priors, values = self.evaluate(boards)
            for (node, path), pri, val in zip(leaves, priors, values):
                node.children = {m: _Node(p) for m, p in pri.items()}
                self._backup(path, float(val))
        return leaves, terminal_backups

    def _select_child(self, node, path):
        total = sum(c.visits for c in node.children.values())
        sqrt_n = (total + 1) ** 0.5
        best, best_score, best_move = None, -1e9, None
        for move, child in node.children.items():
            q = child.value_sum / child.visits if child.visits else 0.0
            score = q + self.c_puct * child.prior * sqrt_n / (1 + child.visits)
            if score > best_score:
                best, best_score, best_move = child, score, move
        # Selection increments the visit and applies virtual loss so other
        # batch members are steered elsewhere until backup settles the value.
        best.visits += 1
        best.value_sum -= _VIRTUAL_LOSS
        path.append((best_move, best))
        return best

    def _backup(self, path, leaf_value):
        # leaf_value is from the leaf mover's perspective; each edge above
        # stores value from ITS mover's perspective, so the sign alternates.
        value = -leaf_value
        for _, node in reversed(path):
            node.value_sum += value + _VIRTUAL_LOSS  # undo virtual loss, add result
            value = -value

    def _undo_virtual(self, path):
        for _, node in path:
            node.visits -= 1
            node.value_sum += _VIRTUAL_LOSS
