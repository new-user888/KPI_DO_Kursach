"""
Генератор задач за псевдокодом п. 3.2.

Параметри:
  m, n    — розмір парку
  rho_F   — частка заборонених клітин (0 ≤ rho_F < 1)
  s_F     — тип структури F:
              1 — розсіяна (випадкова вибірка)
              2 — компактна (BFS-нарощення)
              3 — лінійна (горизонтальна або вертикальна смуга)
              4 — два острови (дві окремі компактні групи)
  c1, c2  — вартості ліхтарів (c2 > c1)
  alpha_B — коефіцієнт бюджету: B = round(alpha_B · |A| · (c1+c2)/2)
  K2      — макс. кількість ліхтарів типу 2
  seed    — зерно генератора випадкових чисел
"""
import random
from typing import Set

from .problem import Cell, Problem


def _grow_compact(start: Cell, target: int, exclude: Set[Cell],
                  m: int, n: int, rng: random.Random) -> Set[Cell]:
    """BFS-нарощення компактної множини від start до target клітин."""
    if target <= 0:
        return set()
    group: Set[Cell] = {start}
    frontier = [start]
    while len(group) < target and frontier:
        v = rng.choice(frontier)
        i, j = v
        neighbours = [
            (i + di, j + dj)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= i + di < m and 0 <= j + dj < n
            and (i + di, j + dj) not in group
            and (i + di, j + dj) not in exclude
        ]
        if not neighbours:
            frontier.remove(v)
        else:
            u = rng.choice(neighbours)
            group.add(u)
            frontier.append(u)
    return group


def generate(m: int, n: int, rho_F: float, s_F: int,
             c1: float, c2: float, alpha_B: float, K2: int,
             seed: int = 42, name: str = "") -> Problem:
    """Генерує задачу із заданими параметрами. seed фіксує результат для повторюваності."""
    rng = random.Random(seed)
    G = [(i, j) for i in range(m) for j in range(n)]
    q_F = max(0, min(round(rho_F * len(G)), len(G) - 1))

    F: Set[Cell] = set()

    if q_F == 0:
        pass

    elif s_F == 1:
        # Розсіяна: випадкова вибірка без повторень
        F = set(rng.sample(G, q_F))

    elif s_F == 2:
        # Компактна: один BFS-острів
        F = _grow_compact(rng.choice(G), q_F, set(), m, n, rng)

    elif s_F == 3:
        # Лінійна: послідовні рядки або стовпці від стартового,
        # відсортовані за відстанню від нього
        if rng.choice([True, False]):
            i0 = rng.randint(0, m - 1)
            for i in sorted(range(m), key=lambda i: abs(i - i0)):
                for j in range(n):
                    F.add((i, j))
                    if len(F) >= q_F:
                        break
                if len(F) >= q_F:
                    break
        else:
            j0 = rng.randint(0, n - 1)
            for j in sorted(range(n), key=lambda j: abs(j - j0)):
                for i in range(m):
                    F.add((i, j))
                    if len(F) >= q_F:
                        break
                if len(F) >= q_F:
                    break

    elif s_F == 4:
        # Два острови: перший BFS-острів, другий — від найвіддаленішої точки.
        # Якщо q_F==1, весь ліміт йде у другий острів (перший порожній).
        half1 = q_F // 2
        half2 = q_F - half1
        group1 = _grow_compact(rng.choice(G), half1, set(), m, n, rng)
        F |= group1
        free = [c for c in G if c not in F]
        if free:
            if group1:
                ci = sum(c[0] for c in group1) // len(group1)
                cj = sum(c[1] for c in group1) // len(group1)
                v1 = max(free, key=lambda c: abs(c[0] - ci) + abs(c[1] - cj))
            else:
                v1 = rng.choice(free)
            F |= _grow_compact(v1, half2, F, m, n, rng)

    else:
        raise ValueError(f"Невідомий тип структури s_F={s_F}")

    B = round(alpha_B * (len(G) - len(F)) * (c1 + c2) / 2)
    return Problem(
        m=m, n=n, F=frozenset(F),
        c1=c1, c2=c2, B=B, K2=K2,
        name=name or f"gen_{m}x{n}_rho{rho_F:.2f}_s{s_F}_seed{seed}",
    )
