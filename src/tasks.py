"""
Чотири індивідуальні задачі з підрозділу 1.2 курсової роботи.
Координати у документі — 1-базовані (i; j), тут перетворюємо в 0-базовані.
"""
from .problem import Problem


def _shift(cells):
    """1-базовані координати → 0-базовані."""
    return frozenset((i - 1, j - 1) for i, j in cells)


def task_kalachuk() -> Problem:
    """8x8, шість окремих клумб."""
    return Problem(
        m=8, n=8,
        F=_shift([(2, 3), (2, 6), (4, 4), (5, 7), (7, 2), (7, 5)]),
        c1=3, c2=7, B=36, K2=3,
        name="Задача №1 (Калачук)",
    )


def task_voloshyn() -> Problem:
    """9x9, водойма 3x3 у центрі."""
    return Problem(
        m=9, n=9,
        F=_shift([(r, c) for r in range(4, 7) for c in range(4, 7)]),
        c1=4, c2=9, B=45, K2=2,
        name="Задача №2 (Волошин)",
    )


def task_korotaiev() -> Problem:
    """10x8, центральна алея — стовпець 4."""
    return Problem(
        m=10, n=8,
        F=_shift([(i, 4) for i in range(1, 11)]),
        c1=3, c2=8, B=40, K2=3,
        name="Задача №3 (Коротаєв)",
    )


def task_danyliuk() -> Problem:
    """6x12, два ставки 2x2."""
    return Problem(
        m=6, n=12,
        F=_shift([(2, 3), (2, 4), (3, 3), (3, 4),
                  (4, 9), (4, 10), (5, 9), (5, 10)]),
        c1=4, c2=10, B=48, K2=3,
        name="Задача №4 (Данилюк)",
    )


def all_tasks() -> list:
    return [task_kalachuk(), task_voloshyn(), task_korotaiev(), task_danyliuk()]
