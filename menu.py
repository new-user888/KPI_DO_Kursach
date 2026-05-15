"""
Консольне меню програми «Розміщення ліхтарів у парку».
Головна точка входу — функція run().
"""
import csv
import json
import os
import random
import sys
import time
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.algorithms import greedy, local_search
from src.experiments import (
    experiment_beta, experiment_budget, experiment_rho,
    experiment_size, experiment_structure,
)
from src.generator import generate
from src.problem import Problem, Solution
from src.tasks import (
    all_tasks, task_kalachuk, task_voloshyn,
    task_korotaiev, task_danyliuk,
)
from src.visualize import (
    plot_park, plot_solution,
    plot_beta_experiment, plot_budget_experiment,
    plot_rho_experiment, plot_size_experiment,
    plot_structure_experiment,
)

_ROOT     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_ROOT, "output")
TASKS_DIR  = os.path.join(OUTPUT_DIR, "tasks")
EXP_DIR    = os.path.join(OUTPUT_DIR, "experiments")

# Глобальний стан сесії
_problem:       Optional[Problem]  = None
_solution:      Optional[Solution] = None
_solution_algo: str                = ""


#  Утиліти вводу / виводу 

def _sep(ch="", width=56):
    print(ch * width)


def _header(title: str):
    print()
    _sep()
    print(f"  {title}")
    _sep()


def _pause():
    input("\nНатисніть Enter для продовження...")


def _read_int(prompt: str, *, min_val: int = None, max_val: int = None) -> int:
    while True:
        try:
            val = int(input(prompt))
        except ValueError:
            print("  Будь ласка, введіть ціле число.")
            continue
        if min_val is not None and val < min_val:
            print(f"  Значення не може бути менше {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"  Значення не може бути більше {max_val}.")
            continue
        return val


def _read_pos_float(prompt: str, *, gt: float = 0.0) -> float:
    while True:
        try:
            val = float(input(prompt))
        except ValueError:
            print("  Будь ласка, введіть число.")
            continue
        if val <= gt:
            print(f"  Значення має бути більше {gt}.")
            continue
        return val


def _choose(max_opt: int) -> int:
    return _read_int("Введіть число: ", min_val=0, max_val=max_opt)


def _ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _write_csv(path: str, rows: list):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


#  Текстова схема парку

def _park_ascii(problem: Problem, solution: Solution = None) -> str:
    """
    Символьна схема парку:
      #  — заборонена клітина
      .  — вільна дозволена
      *  — освітлена дозволена
      1  — ліхтар типу 1
      2  — ліхтар типу 2
    Координати виводяться в 1-базованому вигляді.
    """
    lit    = solution.illuminated() if solution else set()
    lamps  = {cell: t for cell, t in solution.lights} if solution else {}

    lines = []
    col_hdr = "     " + "  ".join(f"{j+1:>2}" for j in range(problem.n))
    lines.append(col_hdr)
    lines.append("    +" + "----" * problem.n)

    for i in range(problem.m):
        row = f" {i+1:>2} |"
        for j in range(problem.n):
            cell = (i, j)
            if cell in problem.F:
                ch = "  # "
            elif cell in lamps:
                ch = f"  {lamps[cell]} "
            elif cell in lit:
                ch = "  * "
            else:
                ch = "  . "
            row += ch
        lines.append(row)

    return "\n".join(lines)


def _legend(with_solution: bool = False):
    print()
    base = "  #-заборонена  .-вільна"
    if with_solution:
        print(base + "  *-освітлена  1/2-ліхтар типу 1/2")
    else:
        print(base)


#  Виведення розв'язку 

def _print_solution(problem: Problem, sol: Solution, algo: str, t_ms: float = None):
    A = len(problem.A)
    print(f"\n  Алгоритм:    {algo}")
    print(f"  Освітлено:   {sol.Q()} з {A}  ({sol.Q()/A*100:.1f}%)")
    print(f"  Вартість:    {int(sol.cost())} з {problem.B}")
    n1 = sum(1 for _, t in sol.lights if t == 1)
    n2 = sol.count_type2()
    print(f"  Ліхтарів:   {len(sol.lights)}  (тип 1: {n1},  тип 2: {n2} з {problem.K2})")
    print(f"  Допустимий: {'так' if sol.is_feasible() else 'НІ — ПОМИЛКА'}")
    if t_ms is not None:
        print(f"  Час роботи:  {t_ms:.2f} мс")
    if sol.lights:
        print()
        print("  Розміщення ліхтарів:")
        for (i, j), t in sol.lights:
            print(f"    рядок {i+1:>2}, стовпець {j+1:>2}  —  ліхтар типу {t}")


def _status_str():
    if _problem is None:
        ps = "задачу не задано"
    else:
        ps = (f"задано  ({_problem.m}×{_problem.n}, "
              f"|A|={len(_problem.A)}, B={_problem.B}, K2={_problem.K2})")
    if _solution is None:
        ss = "відсутній"
    else:
        ss = (f"наявний  ({_solution_algo}, "
              f"Q={_solution.Q()}/{len(_problem.A)})")
    return ps, ss


#  Підменю 1: Внесення даних 

def _input_manual() -> Optional[Problem]:
    _header("Введення задачі вручну")
    try:
        m = _read_int("  Кількість рядків:    ", min_val=1)
        n = _read_int("  Кількість стовпців:  ", min_val=1)

        print()
        print("  Вводьте координати заборонених клітин: рядок стовпець")
        print(f"  Нумерація з 1 (рядки 1..{m}, стовпці 1..{n}).")
        print("  Завершення — порожній рядок.")
        forbidden: set = set()
        while True:
            raw = input("  Клітина: ").strip().replace(",", " ")
            if not raw:
                break
            parts = raw.split()
            if len(parts) != 2:
                print("  Формат: рядок стовпець  (два числа через пробіл).")
                continue
            try:
                r, c = int(parts[0]), int(parts[1])
            except ValueError:
                print("  Введіть два цілих числа.")
                continue
            if not (1 <= r <= m and 1 <= c <= n):
                print(f"  Клітина ({r};{c}) виходить за межі {m}×{n}.")
                continue
            forbidden.add((r - 1, c - 1))

        print()
        c1 = _read_pos_float("  Вартість ліхтаря типу 1 (c1 > 0): ")
        c2 = _read_pos_float(f"  Вартість ліхтаря типу 2 (c2 > {c1}): ", gt=c1)
        B  = _read_pos_float("  Бюджет B (> 0): ")
        K2 = _read_int("  Максимальна кількість ліхтарів типу 2 (K2 ≥ 0): ", min_val=0)

        p = Problem(m=m, n=n, F=frozenset(forbidden),
                    c1=c1, c2=c2, B=B, K2=K2,
                    name="Введена вручну")
        print(f"\n  Задачу створено:  {p.info()}")
        return p

    except ValueError as e:
        print(f"\n  Помилка у даних: {e}")
        return None


def _input_individual() -> Optional[Problem]:
    _header("Вибір індивідуальної задачі")
    options = [
        ("Задача №1 (Калачук)   — 8×8,  6 клумб,       c1=3, c2=7,  B=36, K2=3", task_kalachuk),
        ("Задача №2 (Волошин)   — 9×9,  водойма 3×3,   c1=4, c2=9,  B=45, K2=2", task_voloshyn),
        ("Задача №3 (Коротаєв)  — 10×8, алея стовп.4,  c1=3, c2=8,  B=40, K2=3", task_korotaiev),
        ("Задача №4 (Данилюк)   — 6×12, два ставки 2×2, c1=4, c2=10, B=48, K2=3", task_danyliuk),
    ]
    print()
    for i, (desc, _) in enumerate(options, 1):
        print(f"  {i}. {desc}")
    print("  0. Повернутися")
    ch = _choose(len(options))
    if ch == 0:
        return None
    p = options[ch - 1][1]()
    print(f"\n  Обрано: {p.name}")
    return p


def _input_generate() -> Optional[Problem]:
    _header("Генерація задачі")
    try:
        m = _read_int("  Кількість рядків:   ", min_val=1)
        n = _read_int("  Кількість стовпців: ", min_val=1)
        total = m * n

        raw = input(
            f"  Частка заборонених клітин 0..0.99 "
            f"або кількість 1..{total-1} (Enter — 0.10): "
        ).strip()
        if not raw:
            rho_F = 0.10
        else:
            val = float(raw)
            rho_F = val if 0 < val < 1 else max(0.0, min(val / total, 0.99))
        rho_F = min(rho_F, (total - 1) / total)

        print()
        print("  Структура заборонених зон:")
        print("    1. Випадкові окремі заборонені клітини")
        print("    2. Кілька невеликих клумб (розсіяна)")
        print("    3. Водойма / компактна група")
        print("    4. Центральна алея / смуга")
        print("    5. Дві водойми / дві компактні групи")
        struct_ch = _read_int("  Оберіть тип (1–5): ", min_val=1, max_val=5)
        s_F = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4}[struct_ch]

        print()
        c1 = _read_pos_float("  Вартість ліхтаря типу 1 (c1 > 0): ")
        c2 = _read_pos_float(f"  Вартість ліхтаря типу 2 (c2 > {c1}): ", gt=c1)
        B  = _read_pos_float("  Бюджет B (> 0): ")
        K2 = _read_int("  Максимальна кількість ліхтарів типу 2 (K2 ≥ 0): ", min_val=0)

        seed_raw = input("  Seed (Enter — випадковий): ").strip()
        try:
            seed = int(seed_raw)
        except ValueError:
            seed = random.randint(0, 99999)
            print(f"  Використано seed = {seed}")

        # Генеруємо структуру F, потім підставляємо точний бюджет користувача
        exp_A = max(1, total * (1 - rho_F))
        alpha_B = B / (exp_A * (c1 + c2) / 2)
        p_gen = generate(m=m, n=n, rho_F=rho_F, s_F=s_F,
                         c1=c1, c2=c2, alpha_B=alpha_B, K2=K2, seed=seed)
        p = Problem(m=m, n=n, F=p_gen.F, c1=c1, c2=c2, B=B, K2=K2,
                    name=f"Згенерована {m}×{n} seed={seed}")
        print(f"\n  Задачу згенеровано: {p.info()}")
        return p

    except (ValueError, Exception) as e:
        print(f"\n  Помилка при генерації: {e}")
        return None


def _input_file() -> Optional[Problem]:
    _header("Зчитування задачі з файлу JSON")
    print()
    print('  Очікуваний формат (координати 1-базовані):')
    print('  {')
    print('    "rows": 8,  "cols": 8,')
    print('    "forbidden_cells": [[2,3],[4,4]],')
    print('    "cost_type_1": 3,  "cost_type_2": 7,')
    print('    "budget": 36,  "max_type_2": 3')
    print('  }')
    print()

    path = input("  Шлях до файлу: ").strip()
    if not path:
        print("  Шлях не введено.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m  = int(data["rows"])
        n  = int(data["cols"])
        F  = frozenset(
            (int(r) - 1, int(c) - 1)
            for r, c in data.get("forbidden_cells", [])
        )
        c1 = float(data["cost_type_1"])
        c2 = float(data["cost_type_2"])
        B  = float(data["budget"])
        K2 = int(data["max_type_2"])
        p = Problem(m=m, n=n, F=F, c1=c1, c2=c2, B=B, K2=K2,
                    name=data.get("name", f"З файлу {os.path.basename(path)}"))
        print(f"\n  Зчитано: {p.info()}")
        return p
    except FileNotFoundError:
        print(f"  Файл не знайдено: {path}")
    except json.JSONDecodeError as e:
        print(f"  Невірний формат JSON: {e}")
    except KeyError as e:
        print(f"  Відсутнє поле у файлі: {e}")
    except ValueError as e:
        print(f"  Некоректні дані: {e}")
    return None


def _menu_input():
    global _problem, _solution, _solution_algo
    while True:
        _header("Підменю внесення даних задачі")
        print()
        print("  1. Ввести дані вручну")
        print("  2. Обрати одну з індивідуальних задач")
        print("  3. Згенерувати задачу випадковим чином")
        print("  4. Зчитати задачу з файлу JSON")
        print("  0. Повернутися в головне меню")
        ch = _choose(4)
        if ch == 0:
            return
        result = None
        if ch == 1:
            result = _input_manual()
        elif ch == 2:
            result = _input_individual()
        elif ch == 3:
            result = _input_generate()
        elif ch == 4:
            result = _input_file()
        if result is not None:
            _problem = result
            _solution = None
            _solution_algo = ""
            print("\n  Дані задачі прийнято.")
        _pause()


#  Підменю 2: Розв'язання 

def _run_greedy(problem: Problem, verbose: bool = True):
    t0 = time.perf_counter()
    S, steps = greedy(problem, trace=True)
    t_ms = (time.perf_counter() - t0) * 1000
    if verbose:
        _print_solution(problem, S, "Жадібний алгоритм", t_ms)
        if steps:
            print()
            print("  Покрокове виконання:")
            print("    Крок   Клітина    Тип   Приріст   Ефект-ть    Q")
            for s in steps:
                i, j = s["cell"]
                print(f"     {s['step']:>3}    ({i+1};{j+1})      {s['type']}"
                      f"       {s['gain']:>4}       {s['eff']:>5.2f}    {s['Q']:>3}")
    return S, t_ms


def _ask_imax(problem: Problem) -> int:
    default = 4 * len(problem.A)
    print(f"  Рекомендований I_max = {default}  (4 × |A| = {len(problem.A)})")
    raw = input(f"  Введіть I_max (Enter — {default}): ").strip()
    try:
        val = int(raw)
        return val if val > 0 else default
    except ValueError:
        return default


def _run_local_search(problem: Problem, S_init: Solution,
                      imax: int = None, verbose: bool = True):
    if imax is None:
        imax = 4 * len(problem.A)
    t0 = time.perf_counter()
    S_LS, hist = local_search(problem, S_init, I_max=imax, trace=True)
    t_ms = (time.perf_counter() - t0) * 1000
    if verbose:
        iters = hist[-1]["iter"] if hist else 0
        _print_solution(problem, S_LS, "Локальний пошук", t_ms)
        print(f"  I_max={imax},  виконано ітерацій: {iters}")
        print(f"  Покращення відносно жадібного: {S_LS.Q() - S_init.Q():+d} клітин")
    return S_LS, t_ms


def _save_solution_png(problem, sol, name="last"):
    try:
        _ensure_dirs(TASKS_DIR)
        path = os.path.join(TASKS_DIR, f"{name}.png")
        plot_solution(problem, sol, name.replace("_", " ").title(), path,
                      f"Q={sol.Q()}/{len(problem.A)}  cost={int(sol.cost())}/{problem.B}")
        print(f"  PNG: output/tasks/{name}.png")
    except Exception:
        pass


def _menu_solve():
    global _problem, _solution, _solution_algo
    while True:
        _header("Підменю розв'язання задачі")
        if _problem is None:
            print("\n  Задачу не задано. Спочатку внесіть або згенеруйте дані (пункт 1).")
            _pause()
            return
        print(f"\n  Задача: {_problem.name}  "
              f"({_problem.m}×{_problem.n}, |A|={len(_problem.A)})")
        print()
        print("  1. Розв'язати жадібним алгоритмом")
        print("  2. Розв'язати алгоритмом локального пошуку")
        print("  3. Розв'язати обома алгоритмами")
        print("  0. Повернутися в головне меню")
        ch = _choose(3)
        if ch == 0:
            return
        print()

        if ch == 1:
            S, t = _run_greedy(_problem)
            _solution, _solution_algo = S, "Жадібний"
            _save_solution_png(_problem, S, "last_greedy")

        elif ch == 2:
            S_G, _ = _run_greedy(_problem, verbose=False)
            imax = _ask_imax(_problem)
            S, t = _run_local_search(_problem, S_G, imax=imax)
            _solution, _solution_algo = S, "Локальний пошук"
            _save_solution_png(_problem, S, "last_local_search")

        elif ch == 3:
            print("  Крок 1 — Жадібний алгоритм:")
            S_G, t_G = _run_greedy(_problem)
            _save_solution_png(_problem, S_G, "last_greedy")

            print()
            print("  Крок 2 — Локальний пошук:")
            imax = _ask_imax(_problem)
            S_LS, t_LS = _run_local_search(_problem, S_G, imax=imax)
            _save_solution_png(_problem, S_LS, "last_local_search")

            print()
            best = "локальний пошук" if S_LS.Q() >= S_G.Q() else "жадібний"
            print(f"  Порівняння:   жадібний Q={S_G.Q()},  "
                  f"локальний пошук Q={S_LS.Q()}")
            print(f"  Покращення:   {S_LS.Q() - S_G.Q():+d} клітин")
            print(f"  Кращий результат:  {best}")
            _solution = S_LS
            _solution_algo = "Локальний пошук (після жадібного)"

        _pause()


#  Підменю 3: Експерименти 

def _exp_individual():
    _header("Розв'язання 4 індивідуальних задач")
    _ensure_dirs(TASKS_DIR, EXP_DIR)
    rows = []
    for k, task in enumerate(all_tasks(), 1):
        print(f"\n  Задача {k}: {task.name}  ({task.m}×{task.n}, |A|={len(task.A)})")
        S_G, t_G = _run_greedy(task, verbose=False)
        S_LS, t_LS = _run_local_search(task, S_G, verbose=False)
        imp = S_LS.Q() - S_G.Q()
        print(f"    Жадібний:        Q={S_G.Q()}/{len(task.A)}"
              f"  cost={int(S_G.cost())}/{task.B}"
              f"  t={t_G:.1f}мс")
        print(f"    Локальний пошук: Q={S_LS.Q()}/{len(task.A)}"
              f"  cost={int(S_LS.cost())}/{task.B}"
              f"  t={t_LS:.1f}мс  ({imp:+d})")
        try:
            plot_park(task, f"Задача {k}", os.path.join(TASKS_DIR, f"task{k}_park.png"))
            plot_solution(task, S_G, f"Жадібний (задача {k})",
                          os.path.join(TASKS_DIR, f"task{k}_greedy.png"),
                          f"Q={S_G.Q()}/{len(task.A)}")
            plot_solution(task, S_LS, f"Локальний пошук (задача {k})",
                          os.path.join(TASKS_DIR, f"task{k}_local_search.png"),
                          f"Q={S_LS.Q()}/{len(task.A)}")
        except Exception:
            pass
        rows.append({
            "task": task.name, "m": task.m, "n": task.n,
            "allowed": len(task.A), "budget": task.B, "K2": task.K2,
            "greedy_Q": S_G.Q(), "greedy_cost": int(S_G.cost()),
            "greedy_t_ms": round(t_G, 2),
            "ls_Q": S_LS.Q(), "ls_cost": int(S_LS.cost()),
            "ls_t_ms": round(t_LS, 2), "improvement": imp,
        })
    _write_csv(os.path.join(TASKS_DIR, "individual_tasks.csv"), rows)
    print(f"\n  CSV: output/tasks/individual_tasks.csv")
    print(f"  PNG: output/tasks/task{{1-4}}_*.png")


def _exp_imax():
    _header("Вплив параметра I_max (β · |A|)")
    _ensure_dirs(EXP_DIR)
    m, n, rho_F, s_F = 15, 15, 0.15, 2
    c1, c2, alpha_B, K2, R = 3, 8, 0.05, 3, 3
    betas = [0.01, 0.05, 0.25, 1, 2, 4, 8, 16]
    print(f"  Парк {m}×{n},  ρ_F={rho_F},  c1={c1}, c2={c2},  "
          f"α_B={alpha_B},  K2={K2},  R={R}")
    t0 = time.perf_counter()
    res = experiment_beta(m, n, rho_F, s_F, c1, c2, alpha_B, K2, betas=betas, R=R)
    elapsed = time.perf_counter() - t0
    print(f"\n       β   |   Q (сер ± σ)    |    t, с (сер)")
    print(f"  ---------|-----------------|---------------")
    for b, qm, qs, tm, _ in zip(betas, res["Q_means"], res["Q_stds"],
                                  res["t_means"], res["t_stds"]):
        print(f"  {b:>7}  |  {qm:>6.2f} ± {qs:>5.2f}  |     {tm:>6.3f}")
    try:
        plot_beta_experiment(betas, res["Q_means"], res["t_means"],
                             res["Q_stds"], res["t_stds"],
                             filepath=os.path.join(EXP_DIR, "exp_beta.png"),
                             problem_info=f"{m}×{n}, R={R}")
        print(f"\n  PNG: output/experiments/exp_beta.png")
    except Exception:
        pass
    _write_csv(os.path.join(EXP_DIR, "beta_experiment.csv"), [
        {"beta": b, "Q_mean": round(qm, 4), "Q_std": round(qs, 4),
         "t_mean_s": round(tm, 6)}
        for b, qm, qs, tm, _ in zip(betas, res["Q_means"], res["Q_stds"],
                                     res["t_means"], res["t_stds"])
    ])
    print(f"  CSV: output/experiments/beta_experiment.csv")
    print(f"  Час виконання: {elapsed:.1f} с")


def _exp_budget():
    _header("Вплив бюджету (α_B)")
    _ensure_dirs(EXP_DIR)
    m, n, rho_F, s_F = 14, 14, 0.12, 2
    c1, c2, K2, beta, R = 3, 8, 5, 4, 3
    alphas = [0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25]
    print(f"  Парк {m}×{n},  ρ_F={rho_F},  c1={c1}, c2={c2},  K2={K2},  β={beta},  R={R}")
    t0 = time.perf_counter()
    res = experiment_budget(m, n, rho_F, s_F, c1, c2, K2,
                            alphas=alphas, beta=beta, R=R)
    elapsed = time.perf_counter() - t0
    print(f"\n    α_B   |   B̄  |  Q жадібний  |  Q лок.пошук  |  δ, %")
    print(f"  --------|------|--------------|---------------|--------")
    for a, Bm, qg, qls, dm, _ in zip(alphas, res["budgets_mean"],
                                       res["Q_G_means"], res["Q_LS_means"],
                                       res["delta_means"], res["delta_stds"]):
        print(f"  {a:.2f}    | {Bm:>4.0f} |   {qg:>7.2f}    |    {qls:>7.2f}    | {100*dm:>+5.2f}")
    try:
        plot_budget_experiment(alphas, res["Q_G_means"], res["Q_LS_means"],
                               res["delta_means"], res["delta_stds"],
                               filepath=os.path.join(EXP_DIR, "exp_budget.png"),
                               problem_info=f"{m}×{n}, R={R}")
        print(f"\n  PNG: output/experiments/exp_budget.png")
    except Exception:
        pass
    _write_csv(os.path.join(EXP_DIR, "budget_experiment.csv"), [
        {"alpha_B": a, "budget_mean": round(Bm, 1),
         "Q_G_mean": round(qg, 4), "Q_LS_mean": round(qls, 4),
         "delta_pct": round(100 * dm, 4)}
        for a, Bm, qg, qls, dm, _ in zip(alphas, res["budgets_mean"],
                                           res["Q_G_means"], res["Q_LS_means"],
                                           res["delta_means"], res["delta_stds"])
    ])
    print(f"  CSV: output/experiments/budget_experiment.csv")
    print(f"  Час виконання: {elapsed:.1f} с")


def _exp_size():
    _header("Вплив розмірності парку")
    _ensure_dirs(EXP_DIR)
    sizes = [(6, 6), (8, 8), (10, 10), (12, 12), (14, 14), (16, 16)]
    rho_F, s_F, c1, c2, alpha_B, K2, beta, R = 0.12, 2, 3, 8, 0.06, 3, 4, 3
    print(f"  ρ_F={rho_F},  c1={c1}, c2={c2},  α_B={alpha_B},  K2={K2},  β={beta},  R={R}")
    t0 = time.perf_counter()
    res = experiment_size(sizes, rho_F, s_F, c1, c2, alpha_B, K2, beta=beta, R=R)
    elapsed = time.perf_counter() - t0
    print(f"\n    m×n   |  Q жадібний  |  Q лок.пошук  |  t_G, с  |  t_LS, с  |  δ, %")
    print(f"  --------|--------------|---------------|----------|-----------|-------")
    for (m, n), qg, qls, tg, tls, dm, _ in zip(
            sizes, res["Q_G_means"], res["Q_LS_means"],
            res["t_G_means"], res["t_LS_means"],
            res["delta_means"], res["delta_stds"]):
        print(f"  {m:>2}×{n:<2}   |  {qg:>7.1f}     |   {qls:>7.1f}     | "
              f"{tg:>7.4f}  | {tls:>8.3f}  | {100*dm:>+5.2f}")
    try:
        plot_size_experiment(res["sizes"], res["Q_G_means"], res["Q_LS_means"],
                             res["t_G_means"], res["t_LS_means"],
                             res["delta_means"], res["delta_stds"],
                             filepath=os.path.join(EXP_DIR, "exp_size.png"),
                             problem_info=f"R={R}")
        print(f"\n  PNG: output/experiments/exp_size.png")
    except Exception:
        pass
    _write_csv(os.path.join(EXP_DIR, "size_experiment.csv"), [
        {"m": m, "n": n, "size": m * n,
         "Q_G_mean": round(qg, 4), "Q_LS_mean": round(qls, 4),
         "t_G_s": round(tg, 6), "t_LS_s": round(tls, 6),
         "delta_pct": round(100 * dm, 4)}
        for (m, n), qg, qls, tg, tls, dm, _ in zip(
            sizes, res["Q_G_means"], res["Q_LS_means"],
            res["t_G_means"], res["t_LS_means"],
            res["delta_means"], res["delta_stds"])
    ])
    print(f"  CSV: output/experiments/size_experiment.csv")
    print(f"  Час виконання: {elapsed:.1f} с")


def _exp_rho():
    """Експеримент 3.3.3: вплив частки заборонених клітин rho_F."""
    _header("Вплив частки заборонених клітин (rho_F)")
    _ensure_dirs(EXP_DIR)
    m, n = 12, 12
    rhos = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    s_F, c1, c2, alpha_B, K2, beta, R = 2, 3, 8, 0.06, 3, 4, 5
    print(f"  Парк {m}x{n},  s_F=компактна,  c1={c1}, c2={c2},  "
          f"alpha_B={alpha_B},  K2={K2},  beta={beta},  R={R}")
    print(f"  D = {rhos}")
    t0 = time.perf_counter()
    res = experiment_rho(m, n, rhos, s_F, c1, c2, alpha_B, K2,
                         beta=beta, R=R)
    elapsed = time.perf_counter() - t0
    print(f"\n    rho_F |  Q жадібний  |  Q лок.пошук  |  delta, %  ± sigma, %")
    print(f"  --------|--------------|---------------|---------------------")
    for rho, qg, qls, dm, ds in zip(rhos, res["Q_G_means"],
                                     res["Q_LS_means"],
                                     res["delta_means"], res["delta_stds"]):
        print(f"  {rho:.2f}    |  {qg:>7.2f}     |   {qls:>7.2f}     |  "
              f"{100*dm:>+6.3f} ± {100*ds:.3f}")
    try:
        plot_rho_experiment(rhos, res["Q_G_means"], res["Q_LS_means"],
                            res["delta_means"], res["delta_stds"],
                            filepath=os.path.join(EXP_DIR, "exp_rho.png"),
                            problem_info=f"{m}x{n}, beta={beta}, R={R}")
        print(f"\n  PNG: output/experiments/exp_rho.png")
    except Exception:
        pass
    _write_csv(os.path.join(EXP_DIR, "rho_experiment.csv"), [
        {"rho_F": rho, "Q_G_mean": round(qg, 4),
         "Q_LS_mean": round(qls, 4),
         "delta_pct": round(100 * dm, 4),
         "delta_std_pct": round(100 * ds, 4)}
        for rho, qg, qls, dm, ds in zip(rhos, res["Q_G_means"],
                                         res["Q_LS_means"],
                                         res["delta_means"],
                                         res["delta_stds"])
    ])
    print(f"  CSV: output/experiments/rho_experiment.csv")
    print(f"  Час виконання: {elapsed:.1f} с")


def _exp_structure():
    _header("Вплив структури заборонених зон")
    _ensure_dirs(EXP_DIR)
    m, n, rho_F = 12, 12, 0.12
    c1, c2, alpha_B, K2, beta, R = 3, 8, 0.10, 5, 4, 3
    s_fs = [1, 2, 3, 4]
    print(f"  Парк {m}×{n},  ρ_F={rho_F},  c1={c1}, c2={c2},  "
          f"α_B={alpha_B},  K2={K2},  β={beta},  R={R}")
    t0 = time.perf_counter()
    res = experiment_structure(m, n, rho_F, s_fs, c1, c2, alpha_B, K2,
                               beta=beta, R=R)
    elapsed = time.perf_counter() - t0
    print(f"\n  Структура          |  Q жадібний  |  Q лок.пошук  |  δ, %")
    print(f"  -------------------|--------------|---------------|--------")
    for name, qg, qls, dm, _ in zip(res["names"], res["Q_G_means"],
                                     res["Q_LS_means"],
                                     res["delta_means"], res["delta_stds"]):
        print(f"  {name:<18}  |  {qg:>7.2f}     |   {qls:>7.2f}     | {100*dm:>+5.3f}")
    try:
        plot_structure_experiment(s_fs, res["names"],
                                  res["Q_G_means"], res["Q_LS_means"],
                                  res["delta_means"], res["delta_stds"],
                                  filepath=os.path.join(EXP_DIR, "exp_structure.png"),
                                  problem_info=f"{m}×{n}, R={R}")
        print(f"\n  PNG: output/experiments/exp_structure.png")
    except Exception:
        pass
    _write_csv(os.path.join(EXP_DIR, "structure_experiment.csv"), [
        {"structure": name, "Q_G_mean": round(qg, 4),
         "Q_LS_mean": round(qls, 4), "delta_pct": round(100 * dm, 4)}
        for name, qg, qls, dm, _ in zip(res["names"], res["Q_G_means"],
                                          res["Q_LS_means"],
                                          res["delta_means"], res["delta_stds"])
    ])
    print(f"  CSV: output/experiments/structure_experiment.csv")
    print(f"  Час виконання: {elapsed:.1f} с")


def run_all_experiments():
    """Запустити всі 6 експериментів послідовно (без пауз)."""
    _exp_individual()
    _exp_imax()
    _exp_rho()
    _exp_budget()
    _exp_size()
    _exp_structure()
    print("\n  Всі експерименти завершено.")
    print(f"  CSV та PNG збережено у: output/experiments/ та output/tasks/")


def _menu_experiments():
    while True:
        _header("Підменю проведення експериментів")
        print()
        print("  1. Розв'язати 4 індивідуальні задачі")
        print("  2. Дослідити вплив параметра I_max (β)")
        print("  3. Дослідити вплив частки заборонених клітин ρ_F")
        print("  4. Дослідити вплив бюджету (α_B)")
        print("  5. Дослідити вплив розмірності парку")
        print("  6. Дослідити вплив структури заборонених зон")
        print("  7. Виконати всі експерименти")
        print("  0. Повернутися в головне меню")
        ch = _choose(7)
        if ch == 0:
            return
        if ch == 1:
            _exp_individual()
        elif ch == 2:
            _exp_imax()
        elif ch == 3:
            _exp_rho()
        elif ch == 4:
            _exp_budget()
        elif ch == 5:
            _exp_size()
        elif ch == 6:
            _exp_structure()
        elif ch == 7:
            print("  Увага: виконання займе кілька хвилин.")
            run_all_experiments()
        _pause()


#  Підменю 4: Виведення даних задачі 

def _menu_show_problem():
    global _problem, _solution
    while True:
        _header("Підменю виведення даних задачі")
        if _problem is None:
            print("\n  Задачу не задано.")
            _pause()
            return
        print()
        print("  1. Вивести дані задачі у консоль")
        print("  2. Вивести схему парку у консоль")
        print("  3. Записати дані задачі у файл JSON")
        print("  0. Повернутися в головне меню")
        ch = _choose(3)
        if ch == 0:
            return

        if ch == 1:
            print()
            print(f"  {_problem.info()}")
            print(f"  Дозволених клітин: {len(_problem.A)}")
            print(f"  Заборонених клітин: {len(_problem.F)}")
            if _problem.F:
                forbidden_str = "  ".join(
                    f"({r+1};{c+1})" for r, c in sorted(_problem.F)
                )
                print(f"  Заборонені: {forbidden_str}")

        elif ch == 2:
            sol = _solution if _solution else None
            print()
            print(_park_ascii(_problem, sol))
            _legend(with_solution=sol is not None)

        elif ch == 3:
            path = input("  Шлях (Enter — task.json): ").strip() or "task.json"
            data = {
                "name": _problem.name,
                "rows": _problem.m, "cols": _problem.n,
                "forbidden_cells": [[r + 1, c + 1]
                                    for r, c in sorted(_problem.F)],
                "cost_type_1": _problem.c1,
                "cost_type_2": _problem.c2,
                "budget": _problem.B,
                "max_type_2": _problem.K2,
            }
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  Збережено: {path}")
            except OSError as e:
                print(f"  Помилка запису: {e}")

        _pause()


#  Підменю 5: Виведення останнього розв'язку 

def _menu_show_solution():
    global _problem, _solution, _solution_algo
    while True:
        _header("Підменю виведення останнього розв'язку")
        if _solution is None:
            print("\n  Останній розв'язок відсутній. Спочатку розв'яжіть задачу (пункт 2).")
            _pause()
            return
        print()
        print("  1. Вивести розв'язок у консоль")
        print("  2. Вивести схему парку з розв'язком")
        print("  3. Записати розв'язок у файл JSON")
        print("  0. Повернутися в головне меню")
        ch = _choose(3)
        if ch == 0:
            return

        if ch == 1:
            print()
            _print_solution(_problem, _solution, _solution_algo)

        elif ch == 2:
            print()
            print(_park_ascii(_problem, _solution))
            _legend(with_solution=True)

        elif ch == 3:
            path = (input("  Шлях (Enter — solution.json): ").strip()
                    or "solution.json")
            A = len(_problem.A)
            data = {
                "algorithm": _solution_algo,
                "Q": _solution.Q(),
                "Q_total": A,
                "coverage_pct": round(_solution.Q() / A * 100, 2) if A else 0,
                "cost": int(_solution.cost()),
                "budget": _problem.B,
                "type1_count": sum(1 for _, t in _solution.lights if t == 1),
                "type2_count": _solution.count_type2(),
                "feasible": _solution.is_feasible(),
                "lamps": [
                    {"row": i + 1, "col": j + 1, "type": t}
                    for (i, j), t in _solution.lights
                ],
            }
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  Збережено: {path}")
            except OSError as e:
                print(f"  Помилка запису: {e}")

        _pause()


#  Головне меню 

def _main_menu() -> int:
    p_status, s_status = _status_str()
    print()
    _sep("═")
    print("  Розміщення ліхтарів у парку")
    print("  Головне меню")
    _sep("═")
    print(f"  Статус задачі:      {p_status}")
    print(f"  Останній розв'язок: {s_status}")
    _sep()
    print()
    print("  1. Внести дані задачі")
    print("  2. Розв'язати задачу")
    print("  3. Провести експерименти")
    print("  4. Вивести дані задачі")
    print("  5. Вивести останній розв'язок")
    print("  0. Завершити роботу")
    return _choose(5)


def run():
    """Головна точка входу — запускає консольне меню."""
    while True:
        ch = _main_menu()
        if ch == 0:
            print("\n  До побачення!")
            break
        elif ch == 1:
            _menu_input()
        elif ch == 2:
            _menu_solve()
        elif ch == 3:
            _menu_experiments()
        elif ch == 4:
            _menu_show_problem()
        elif ch == 5:
            _menu_show_solution()
