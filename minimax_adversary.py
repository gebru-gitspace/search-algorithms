import random
from typing import List, Any, Callable, Optional, Tuple

# ===============================
# Base game interface
# ===============================

class Game:
    """Abstract base class for a 2-player deterministic, perfect-information, zero-sum game."""
    def actions(self, state: Any) -> List[Any]:
        raise NotImplementedError

    def result(self, state: Any, action: Any) -> Any:
        raise NotImplementedError

    def terminal_test(self, state: Any) -> bool:
        raise NotImplementedError

    def utility(self, state: Any, player: Any) -> float:
        raise NotImplementedError

    def to_move(self, state: Any) -> Any:
        raise NotImplementedError


# ===============================
# Adversary strategy interface
# ===============================

class Adversary:
    """Defines how the opponent (adversary) chooses its move."""
    def choose_action(self, state: Any, actions: List[Any], game: Game, evaluator: Callable[[Any], float], opponent: Any) -> Any:
        raise NotImplementedError


class OptimalAdversary(Adversary):
    """Opponent that plays optimally (minimizes the maximizer’s score)."""
    def choose_action(self, state, actions, game, evaluator, opponent):
        best_action = None
        best_value = float('inf')
        for a in actions:
            val = evaluator(game.result(state, a))
            if val < best_value:
                best_value = val
                best_action = a
        return best_action


class RandomAdversary(Adversary):
    """Opponent that plays randomly."""
    def choose_action(self, state, actions, game, evaluator, opponent):
        return random.choice(actions)


class HeuristicAdversary(Adversary):
    """Opponent that plays based on a heuristic."""
    def __init__(self, heuristic: Callable[[Any], float]):
        self.heuristic = heuristic

    def choose_action(self, state, actions, game, evaluator, opponent):
        best_action = None
        best_value = float('inf')
        for a in actions:
            val = self.heuristic(game.result(state, a))
            if val < best_value:
                best_value = val
                best_action = a
        return best_action


# ===============================
# Minimax algorithm with adversary
# ===============================

def minimax_decision(state: Any, game: Game, player: Any, adversary: Optional[Adversary] = None) -> Any:
    """Return the best action for 'player' in the given state using minimax.
       If an adversary is provided, use it to pick opponent moves instead of assuming optimal play.
    """

    def max_value(s):
        if game.terminal_test(s):
            return game.utility(s, player)
        v = float('-inf')
        for a in game.actions(s):
            v = max(v, min_value(game.result(s, a)))
        return v

    def min_value(s):
        if game.terminal_test(s):
            return game.utility(s, player)
        actions = game.actions(s)
        if adversary is None:
            v = float('inf')
            for a in actions:
                v = min(v, max_value(game.result(s, a)))
            return v
        else:
            chosen = adversary.choose_action(s, actions, game, lambda st: max_value(st), game.to_move(s))
            return max_value(game.result(s, chosen))

    best_action = None
    best_value = float('-inf')
    for a in game.actions(state):
        val = min_value(game.result(state, a))
        if val > best_value:
            best_value = val
            best_action = a
    return best_action


# ===============================
# Example Game: Nim
# ===============================

class NimGame(Game):
    """Simple Nim game with one pile."""
    def __init__(self, max_take: int = 3):
        self.max_take = max_take

    def actions(self, state: Tuple[int, str]) -> List[int]:
        tokens, _ = state
        return [i for i in range(1, min(self.max_take, tokens) + 1)]

    def result(self, state: Tuple[int, str], action: int) -> Tuple[int, str]:
        tokens, to_move = state
        next_player = 'B' if to_move == 'A' else 'A'
        return (tokens - action, next_player)

    def terminal_test(self, state: Tuple[int, str]) -> bool:
        tokens, _ = state
        return tokens == 0

    def utility(self, state: Tuple[int, str], player: str) -> float:
        tokens, to_move = state
        if tokens != 0:
            raise ValueError("Utility should only be called on terminal states.")
        # The player who cannot move loses
        loser = to_move
        winner = 'B' if loser == 'A' else 'A'
        return 1.0 if winner == player else -1.0

    def to_move(self, state: Tuple[int, str]) -> str:
        return state[1]


# ===============================
# Demo
# ===============================

def demo():
    game = NimGame(max_take=3)
    initial_state = (7, 'A')  # 7 tokens, player A to move

    print("Initial state:", initial_state)
    print("Available actions:", game.actions(initial_state))

    # Case 1: standard minimax (opponent plays optimally)
    best_opt = minimax_decision(initial_state, game, 'A')
    print("Best move vs optimal opponent:", best_opt)

    # Case 2: random adversary
    rand_adv = RandomAdversary()
    best_rand = minimax_decision(initial_state, game, 'A', adversary=rand_adv)
    print("Best move vs random opponent:", best_rand)

    # Case 3: heuristic adversary (greedy opponent)
    heur_adv = HeuristicAdversary(lambda s: s[0])  # prefers fewer tokens
    best_heur = minimax_decision(initial_state, game, 'A', adversary=heur_adv)
    print("Best move vs greedy opponent:", best_heur)

    # Case 4: explicit OptimalAdversary (same as standard minimax)
    opt_adv = OptimalAdversary()
    best_opt_adv = minimax_decision(initial_state, game, 'A', adversary=opt_adv)
    print("Best move vs OptimalAdversary:", best_opt_adv)


if __name__ == "__main__":
    demo()
