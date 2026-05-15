"""
Модель задачі розміщення ліхтарів у парку.

Клітини — кортежі (i, j) з 0-базованою індексацією (i ∈ 0..m-1, j ∈ 0..n-1).
У документі використовується 1-базована індексація; при виводі додаємо +1.
"""
from dataclasses import dataclass, field
from typing import FrozenSet, Set, Tuple

Cell = Tuple[int, int]


@dataclass
class Problem:
    """Задача розміщення ліхтарів."""
    m: int              # кількість рядків
    n: int              # кількість стовпців
    F: FrozenSet[Cell]  # заборонені клітини
    c1: float           # вартість ліхтаря типу 1
    c2: float           # вартість ліхтаря типу 2  (c2 > c1)
    B: float            # бюджет
    K2: int             # макс. кількість ліхтарів типу 2
    name: str = ""

    G: FrozenSet[Cell] = field(init=False)  # всі клітини
    A: FrozenSet[Cell] = field(init=False)  # дозволені клітини (G \ F)

    def __post_init__(self):
        self.G = frozenset((i, j) for i in range(self.m) for j in range(self.n))
        self.A = self.G - self.F
        if self.c2 <= self.c1:
            raise ValueError(f"c2 ({self.c2}) має бути більше c1 ({self.c1})")
        for i, j in self.F:
            if not (0 <= i < self.m and 0 <= j < self.n):
                raise ValueError(
                    f"Заборонена клітина ({i},{j}) поза межами {self.m}x{self.n}"
                )

    def coverage(self, u: Cell, t: int) -> Set[Cell]:
        """Клітини парку, які освітлює ліхтар типу t у клітині u.
        Тип 1 дає зону 3×3 (відстань Чебишева ≤ 1), тип 2 — зону 5×5 (≤ 2).
        Зона обрізається по межах парку."""
        i, j = u
        r = 1 if t == 1 else 2
        return {
            (i + di, j + dj)
            for di in range(-r, r + 1)
            for dj in range(-r, r + 1)
            if 0 <= i + di < self.m and 0 <= j + dj < self.n
        }

    def cost_of(self, t: int) -> float:
        return self.c1 if t == 1 else self.c2

    def info(self) -> str:
        return (
            f"Задача '{self.name}': парк {self.m}x{self.n}, "
            f"|G|={len(self.G)}, |F|={len(self.F)}, |A|={len(self.A)}, "
            f"c1={self.c1}, c2={self.c2}, B={self.B}, K2={self.K2}"
        )


@dataclass
class Solution:
    """Розв'язок: список пар (клітина, тип ліхтаря)."""
    lights: list = field(default_factory=list)  # [(cell, type), ...]
    problem: Problem = None

    def cost(self) -> float:
        return sum(self.problem.cost_of(t) for _, t in self.lights)

    def count_type2(self) -> int:
        return sum(1 for _, t in self.lights if t == 2)

    def occupied_cells(self) -> Set[Cell]:
        return {u for u, _ in self.lights}

    def illuminated(self) -> Set[Cell]:
        """Дозволені клітини, освітлені хоча б одним ліхтарем."""
        lit: Set[Cell] = set()
        for u, t in self.lights:
            lit |= self.problem.coverage(u, t) & self.problem.A
        return lit

    def Q(self) -> int:
        """Цільова функція — кількість освітлених дозволених клітин."""
        return len(self.illuminated())

    def is_feasible(self) -> bool:
        p = self.problem
        if self.cost() > p.B + 1e-9:
            return False
        if self.count_type2() > p.K2:
            return False
        cells = [u for u, _ in self.lights]
        if len(set(cells)) != len(cells):
            return False
        return all(u in p.A for u, _ in self.lights)

    def copy(self) -> "Solution":
        return Solution(list(self.lights), self.problem)

    def __repr__(self) -> str:
        parts = [f"({i+1};{j+1})/Л{t}" for (i, j), t in self.lights]
        return f"Solution[Q={self.Q()}, cost={self.cost():.0f}, {parts}]"
