from typing import Any, Tuple, List

# ===============================
# Base Game Interface
# ===============================

class Game:
    """Abstract two-player, deterministic, perfect-information, zero-sum game."""
    def actions(self, state: Any) -> List[Any]:
        raise NotImplementedError

    def result(self, state: Any, action: Any) -> Any:
        raise NotImplementedError

    def is_terminal(self, state: Any) -> bool:
        raise NotImplementedError

    def utility(self, state: Any, player: Any) -> float:
        raise NotImplementedError

    def to_move(self, state: Any) -> Any:
        raise NotImplementedError


# ===============================
# Alpha–Beta Pruning Search
# ===============================

def alpha_beta_search(game: Game, state: Any) -> Any:
    """Return the best action for the player using alpha-beta pruning."""
    player = game.to_move(state)
    value, move = max_value(game, state, player, alpha=float('-inf'), beta=float('inf'))
    return move


def max_value(game: Game, state: Any, player: Any, alpha: float, beta: float) -> Tuple[float, Any]:
    """Return (utility, move) for the maximizing player, with pruning."""
    if game.is_terminal(state):
        return game.utility(state, player), None

    v = float('-inf')
    best_move = None

    for a in game.actions(state):
        v2, _ = min_value(game, game.result(state, a), player, alpha, beta)
        if v2 > v:
            v, best_move = v2, a
        if v >= beta:  # prune
            return v, best_move
        alpha = max(alpha, v)
    return v, best_move


def min_value(game: Game, state: Any, player: Any, alpha: float, beta: float) -> Tuple[float, Any]:
    """Return (utility, move) for the minimizing player, with pruning."""
    if game.is_terminal(state):
        return game.utility(state, player), None

    v = float('inf')
    best_move = None

    for a in game.actions(state):
        v2, _ = max_value(game, game.result(state, a), player, alpha, beta)
        if v2 < v:
            v, best_move = v2, a
        if v <= alpha:  # prune
            return v, best_move
        beta = min(beta, v)
    return v, best_move


# ===============================
# Example Game: Nim
# ===============================

class NimGame(Game):
    """Nim game: one pile of tokens, players take 1–3 tokens per turn."""
    def __init__(self, max_take=3):
        self.max_take = max_take

    def actions(self, state: Tuple[int, str]) -> List[int]:
        tokens, _ = state
        return [i for i in range(1, min(self.max_take, tokens) + 1)]

    def result(self, state: Tuple[int, str], action: int) -> Tuple[int, str]:
        tokens, to_move = state
        next_player = 'B' if to_move == 'A' else 'A'
        return (tokens - action, next_player)

    def is_terminal(self, state: Tuple[int, str]) -> bool:
        tokens, _ = state
        return tokens == 0

    def utility(self, state: Tuple[int, str], player: str) -> float:
        tokens, to_move = state
        if tokens != 0:
            raise ValueError("Utility should only be called on terminal states.")
        # The player who cannot move loses
        loser = to_move
        winner = 'B' if loser == 'A' else 'A'
        return 1 if winner == player else -1

    def to_move(self, state: Tuple[int, str]) -> str:
        return state[1]


# ===============================
# Demo
# ===============================

if __name__ == "__main__":
    game = NimGame(max_take=3)
    state = (7, 'A')  # 7 tokens, player A to move

    best_move = alpha_beta_search(game, state)
    print(f"Best action for player {game.to_move(state)} with {state[0]} tokens: {best_move}")
