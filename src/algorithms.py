"""
Жадібний алгоритм і алгоритм локального пошуку для задачі
розміщення ліхтарів у парку (псевдокоди 2.2.1 і 2.3.1).
"""
import time
from typing import List, Optional, Tuple

from .problem import Cell, Problem, Solution


def greedy(problem: Problem, trace: bool = False) -> Tuple[Solution, List[dict]]:
    """
    Жадібний алгоритм. На кожному кроці обирає пару (клітина, тип)
    з найбільшим відношенням eff = gain / cost, де gain — кількість
    нових освітлених дозволених клітин. При рівній eff обирає варіант
    з більшим gain, а потім з меншою вартістю.
    """
    p = problem
    S = Solution(lights=[], problem=p)
    lit: set = set()        # вже освітлені клітини
    used_budget = 0.0
    used_type2 = 0
    occupied: set = set()
    steps: List[dict] = []

    while True:
        best_cell: Optional[Cell] = None
        best_type: Optional[int] = None
        best_eff = 0.0
        best_gain = 0
        best_new: set = set()
        best_cost = 0.0

        for u in sorted(p.A):
            if u in occupied:
                continue
            for t in (1, 2):
                cost = p.cost_of(t)
                if used_budget + cost > p.B + 1e-9:
                    continue
                if t == 2 and used_type2 >= p.K2:
                    continue
                new_cells = (p.coverage(u, t) & p.A) - lit
                gain = len(new_cells)
                if gain == 0:
                    continue
                eff = gain / cost
                if (eff > best_eff + 1e-12
                        or (abs(eff - best_eff) < 1e-12 and gain > best_gain)
                        or (abs(eff - best_eff) < 1e-12 and gain == best_gain
                            and cost < best_cost)):
                    best_eff, best_gain = eff, gain
                    best_cell, best_type = u, t
                    best_new, best_cost = new_cells, cost

        if best_cell is None:
            break

        S.lights.append((best_cell, best_type))
        lit |= best_new
        used_budget += best_cost
        if best_type == 2:
            used_type2 += 1
        occupied.add(best_cell)

        if trace:
            steps.append({
                "step": len(steps) + 1,
                "cell": best_cell,
                "type": best_type,
                "gain": best_gain,
                "cost": best_cost,
                "eff": best_eff,
                "Q": len(lit),
                "used_budget": used_budget,
                "used_type2": used_type2,
            })

    return S, steps


def _neighbors(S: Solution):
    """
    Генератор усіх сусідніх розв'язків N(S). Підтримує п'ять операторів:
    додавання, видалення, переміщення, заміна типу, заміна у новій клітині.
    """
    p = S.problem
    occupied = S.occupied_cells()
    free = sorted(u for u in p.A if u not in occupied)

    for u in free:                            # додавання
        for t in (1, 2):
            new = S.copy()
            new.lights.append((u, t))
            yield new, "add"

    for k in range(len(S.lights)):            # видалення
        new = S.copy()
        del new.lights[k]
        yield new, "remove"

    for k, (_, t) in enumerate(S.lights):    # переміщення (той самий тип)
        for u in free:
            new = S.copy()
            new.lights[k] = (u, t)
            yield new, "move"

    for k, (cell, t) in enumerate(S.lights): # заміна типу (1 ↔ 2)
        new = S.copy()
        new.lights[k] = (cell, 2 if t == 1 else 1)
        yield new, "swap_type"

    for k in range(len(S.lights)):            # заміна ліхтаря у новій клітині
        for u in free:
            for t in (1, 2):
                new = S.copy()
                new.lights[k] = (u, t)
                yield new, "replace_new"


def local_search(problem: Problem,
                 initial: Solution,
                 I_max: int,
                 trace: bool = False) -> Tuple[Solution, List[dict]]:
    """
    Локальний пошук. Стартує з initial і переходить до найкращого сусіда
    за Q (при рівному Q — за меншою вартістю). Зупиняється, якщо досягнуто
    I_max ітерацій, немає покращення або освітлено всі дозволені клітини.
    """
    p = problem
    S_best = initial.copy()
    if not S_best.is_feasible():
        raise ValueError("Початковий розв'язок не є допустимим")

    Q_best = S_best.Q()
    cost_best = S_best.cost()
    history: List[dict] = []

    if trace:
        history.append({
            "iter": 0, "Q": Q_best, "cost": cost_best,
            "operator": "initial", "lights": len(S_best.lights),
        })

    for iteration in range(1, I_max + 1):
        best_nb: Optional[Solution] = None
        best_nb_Q = Q_best
        best_nb_cost = cost_best
        best_op = None

        for S_prime, op in _neighbors(S_best):
            if not S_prime.is_feasible():
                continue
            Q_prime = S_prime.Q()
            cost_prime = S_prime.cost()
            if (Q_prime > best_nb_Q
                    or (Q_prime == best_nb_Q and cost_prime < best_nb_cost)):
                best_nb, best_nb_Q = S_prime, Q_prime
                best_nb_cost, best_op = cost_prime, op

        if best_nb is None:
            break

        S_best = best_nb
        Q_best, cost_best = best_nb_Q, best_nb_cost

        if trace:
            history.append({
                "iter": iteration, "Q": Q_best, "cost": cost_best,
                "operator": best_op, "lights": len(S_best.lights),
            })

        if Q_best == len(p.A):
            break

    return S_best, history


def solve_greedy_timed(problem: Problem):
    t0 = time.perf_counter()
    S, _ = greedy(problem)
    return S, time.perf_counter() - t0


def solve_local_search_timed(problem: Problem, beta: float = 4.0):
    """Запускає жадібний, а потім локальний пошук з I_max = β·|A|."""
    S_G, t_G = solve_greedy_timed(problem)
    I_max = max(1, int(beta * len(problem.A)))
    t0 = time.perf_counter()
    S_LS, _ = local_search(problem, S_G, I_max)
    return S_LS, time.perf_counter() - t0, S_G, t_G
