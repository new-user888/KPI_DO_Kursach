"""
Шість експериментів для порівняння жадібного алгоритму і локального пошуку
(підрозділи 3.3.2–3.3.6 курсової роботи).
"""
import statistics
import time
from typing import List, Tuple

from .algorithms import greedy, local_search, solve_local_search_timed
from .generator import generate
from .problem import Problem

STRUCTURE_NAMES = {1: "розсіяна", 2: "компактна", 3: "лінійна", 4: "два острови"}


def _run_pair(p: Problem, beta: float) -> Tuple[float, float, float, float, float]:
    """Запускає жадібний і ЛП для однієї задачі, повертає (Qg, Qls, t_G, t_LS, delta)."""
    S_LS, t_LS, S_G, t_G = solve_local_search_timed(p, beta=beta)
    Qg, Qls = S_G.Q(), S_LS.Q()
    delta = (Qls - Qg) / Qg if Qg > 0 else 0.0
    return Qg, Qls, t_G, t_LS, delta


def _agg(vals: list) -> Tuple[float, float]:
    """Повертає (середнє, стандартне відхилення) списку значень."""
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def experiment_beta(m: int, n: int, rho_F: float, s_F: int,
                    c1: float, c2: float, alpha_B: float, K2: int,
                    betas: List[float], R: int, seed_base: int = 1000):
    """
    Для кожного значення β генерує R задач і запускає жадібний + ЛП з I_max = β·|A|.
    Повертає усереднені Q і час роботи для кожного β.
    """
    Q_runs = {b: [] for b in betas}
    t_runs = {b: [] for b in betas}

    for i in range(R):
        p = generate(m, n, rho_F, s_F, c1, c2, alpha_B, K2, seed=seed_base + i)
        S_G, _ = greedy(p)
        for b in betas:
            I_max = max(1, int(b * len(p.A)))
            t0 = time.perf_counter()
            S_LS, _ = local_search(p, S_G, I_max)
            Q_runs[b].append(S_LS.Q())
            t_runs[b].append(time.perf_counter() - t0)

    Q_means, Q_stds = zip(*[_agg(Q_runs[b]) for b in betas])
    t_means, t_stds = zip(*[_agg(t_runs[b]) for b in betas])
    return {
        "betas": list(betas),
        "Q_means": list(Q_means), "Q_stds": list(Q_stds),
        "t_means": list(t_means), "t_stds": list(t_stds),
        "Q_runs": Q_runs, "t_runs": t_runs,
    }


def experiment_rho(m: int, n: int, rhos: List[float], s_F: int,
                   c1: float, c2: float, alpha_B: float, K2: int,
                   beta: float, R: int, seed_base: int = 2000):
    Q_G_means, Q_LS_means, delta_means, delta_stds, raw = [], [], [], [], []

    for rho in rhos:
        runs = [
            _run_pair(generate(m, n, rho, s_F, c1, c2, alpha_B, K2,
                               seed=seed_base + int(rho * 1000) + i), beta)
            for i in range(R)
        ]
        Qg_list   = [r[0] for r in runs]
        Qls_list  = [r[1] for r in runs]
        delta_list = [r[4] for r in runs]

        Q_G_means.append(_agg(Qg_list)[0])
        Q_LS_means.append(_agg(Qls_list)[0])
        dm, ds = _agg(delta_list)
        delta_means.append(dm); delta_stds.append(ds)
        raw.append({"rho": rho, "Qg": Qg_list, "Qls": Qls_list, "delta": delta_list})

    return {"rhos": rhos, "Q_G_means": Q_G_means, "Q_LS_means": Q_LS_means,
            "delta_means": delta_means, "delta_stds": delta_stds, "raw": raw}


def experiment_size(sizes, rho_F: float, s_F: int,
                    c1: float, c2: float, alpha_B: float, K2: int,
                    beta: float, R: int, seed_base: int = 3000):
    Q_G_means, Q_LS_means = [], []
    t_G_means, t_LS_means = [], []
    delta_means, delta_stds, raw = [], [], []

    for (m, n) in sizes:
        runs = [
            _run_pair(generate(m, n, rho_F, s_F, c1, c2, alpha_B, K2,
                               seed=seed_base + m * 100 + n + i), beta)
            for i in range(R)
        ]
        Qg_list   = [r[0] for r in runs]
        Qls_list  = [r[1] for r in runs]
        tg_list   = [r[2] for r in runs]
        tls_list  = [r[3] for r in runs]
        delta_list = [r[4] for r in runs]

        Q_G_means.append(_agg(Qg_list)[0])
        Q_LS_means.append(_agg(Qls_list)[0])
        t_G_means.append(_agg(tg_list)[0])
        t_LS_means.append(_agg(tls_list)[0])
        dm, ds = _agg(delta_list)
        delta_means.append(dm); delta_stds.append(ds)
        raw.append({"size": (m, n), "Qg": Qg_list, "Qls": Qls_list,
                    "tg": tg_list, "tls": tls_list, "delta": delta_list})

    return {
        "sizes": [m * n for (m, n) in sizes], "sizes_mn": sizes,
        "Q_G_means": Q_G_means, "Q_LS_means": Q_LS_means,
        "t_G_means": t_G_means, "t_LS_means": t_LS_means,
        "delta_means": delta_means, "delta_stds": delta_stds, "raw": raw,
    }


def experiment_budget(m: int, n: int, rho_F: float, s_F: int,
                      c1: float, c2: float, K2: int,
                      alphas: List[float], beta: float, R: int,
                      seed_base: int = 5000):
    """B = round(alpha_B · |A| · (c1+c2)/2) для кожного alpha_B."""
    Q_G_means, Q_LS_means, delta_means, delta_stds = [], [], [], []
    budgets_mean, raw = [], []

    for alpha_B in alphas:
        Qg_list, Qls_list, delta_list, B_list = [], [], [], []
        for i in range(R):
            p = generate(m, n, rho_F, s_F, c1, c2, alpha_B, K2,
                         seed=seed_base + int(alpha_B * 10000) + i)
            Qg, Qls, _, _, delta = _run_pair(p, beta)
            Qg_list.append(Qg); Qls_list.append(Qls)
            delta_list.append(delta); B_list.append(p.B)

        Q_G_means.append(_agg(Qg_list)[0])
        Q_LS_means.append(_agg(Qls_list)[0])
        dm, ds = _agg(delta_list)
        delta_means.append(dm); delta_stds.append(ds)
        budgets_mean.append(_agg(B_list)[0])
        raw.append({"alpha_B": alpha_B, "Qg": Qg_list, "Qls": Qls_list,
                    "delta": delta_list, "B": B_list})

    return {"alphas": alphas, "budgets_mean": budgets_mean,
            "Q_G_means": Q_G_means, "Q_LS_means": Q_LS_means,
            "delta_means": delta_means, "delta_stds": delta_stds, "raw": raw}


def experiment_structure(m: int, n: int, rho_F: float,
                         s_fs: List[int],
                         c1: float, c2: float, alpha_B: float, K2: int,
                         beta: float, R: int, seed_base: int = 6000):
    Q_G_means, Q_LS_means, delta_means, delta_stds, raw = [], [], [], [], []

    for s_F in s_fs:
        runs = [
            _run_pair(generate(m, n, rho_F, s_F, c1, c2, alpha_B, K2,
                               seed=seed_base + s_F * 100 + i), beta)
            for i in range(R)
        ]
        Qg_list   = [r[0] for r in runs]
        Qls_list  = [r[1] for r in runs]
        delta_list = [r[4] for r in runs]

        Q_G_means.append(_agg(Qg_list)[0])
        Q_LS_means.append(_agg(Qls_list)[0])
        dm, ds = _agg(delta_list)
        delta_means.append(dm); delta_stds.append(ds)
        raw.append({"s_F": s_F, "name": STRUCTURE_NAMES.get(s_F, str(s_F)),
                    "Qg": Qg_list, "Qls": Qls_list, "delta": delta_list})

    return {
        "s_fs": s_fs,
        "names": [STRUCTURE_NAMES.get(s, str(s)) for s in s_fs],
        "Q_G_means": Q_G_means, "Q_LS_means": Q_LS_means,
        "delta_means": delta_means, "delta_stds": delta_stds, "raw": raw,
    }
