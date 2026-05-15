"""
Функції для побудови рисунків (matplotlib): план парку, план освітлення
та графіки результатів експериментів. Координати відображаються в 1-базованому
вигляді, як у документі.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .problem import Problem, Solution

COLOR_ALLOWED  = "#f5f5f5"
COLOR_FORBIDDEN = "#7a8b9e"
COLOR_ILLUM    = "#fff4b8"
COLOR_LIGHT1   = "#ffb347"
COLOR_LIGHT2   = "#e74c3c"
COLOR_GRID     = "#aaaaaa"


def _draw_grid(ax, m, n):
    """Малювання сітки та координатних підписів (1-базовані)."""
    for i in range(m + 1):
        ax.axhline(i, color=COLOR_GRID, linewidth=0.5)
    for j in range(n + 1):
        ax.axvline(j, color=COLOR_GRID, linewidth=0.5)
    ax.set_xlim(0, n)
    ax.set_ylim(0, m)
    ax.invert_yaxis()
    ax.set_xticks([j + 0.5 for j in range(n)])
    ax.set_xticklabels([str(j + 1) for j in range(n)], fontsize=9)
    ax.set_yticks([i + 0.5 for i in range(m)])
    ax.set_yticklabels([str(i + 1) for i in range(m)], fontsize=9)
    ax.tick_params(length=0)
    ax.set_aspect("equal")
    ax.xaxis.tick_top()


def _fill_cell(ax, i, j, color):
    rect = patches.Rectangle((j, i), 1, 1, linewidth=0,
                             facecolor=color, edgecolor="none")
    ax.add_patch(rect)


def _text_in_cell(ax, i, j, text, color="black", fontsize=11, weight="bold"):
    ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
            color=color, fontsize=fontsize, fontweight=weight)


def plot_park(problem: Problem, title: str, filepath: str):
    """Рисунок: план парку без розв'язку (показано тільки F та A)."""
    fig, ax = plt.subplots(figsize=(0.6 * problem.n + 2, 0.6 * problem.m + 1.5))
    for i in range(problem.m):
        for j in range(problem.n):
            if (i, j) in problem.F:
                _fill_cell(ax, i, j, COLOR_FORBIDDEN)
                _text_in_cell(ax, i, j, "З", color="white")
            else:
                _fill_cell(ax, i, j, COLOR_ALLOWED)
    _draw_grid(ax, problem.m, problem.n)
    ax.set_title(title, fontsize=12, pad=18)
    plt.tight_layout()
    plt.savefig(filepath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_solution(problem: Problem, solution: Solution,
                  title: str, filepath: str, subtitle: str = ""):
    """Малює план освітлення: заборонені (З), ліхтарі (Л1/Л2),
    освітлені дозволені (О) та неосвітлені дозволені клітини."""
    L = solution.illuminated()
    lights_dict = {cell: t for cell, t in solution.lights}

    fig, ax = plt.subplots(figsize=(0.6 * problem.n + 2, 0.6 * problem.m + 2))
    for i in range(problem.m):
        for j in range(problem.n):
            c = (i, j)
            if c in problem.F:
                _fill_cell(ax, i, j, COLOR_FORBIDDEN)
                _text_in_cell(ax, i, j, "З", color="white")
            elif c in lights_dict:
                t = lights_dict[c]
                _fill_cell(ax, i, j, COLOR_LIGHT2 if t == 2 else COLOR_LIGHT1)
                _text_in_cell(ax, i, j, f"Л{t}", color="white")
            elif c in L:
                _fill_cell(ax, i, j, COLOR_ILLUM)
                _text_in_cell(ax, i, j, "О", color="#444", fontsize=10)
            else:
                _fill_cell(ax, i, j, COLOR_ALLOWED)

    _draw_grid(ax, problem.m, problem.n)

    full_title = title
    if subtitle:
        full_title += "\n" + subtitle
    ax.set_title(full_title, fontsize=11, pad=18)
    plt.tight_layout()
    plt.savefig(filepath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_beta_experiment(betas, Q_means, t_means,
                         Q_stds, t_stds,
                         filepath: str,
                         problem_info: str = ""):
    """Графіки Q̄(β) та t̄(β) для експерименту 3.3.2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.errorbar(betas, Q_means, yerr=Q_stds,
                 marker="o", color="#2980b9", linewidth=2, capsize=4)
    ax1.set_xlabel(r"$\beta$ (множник для $I_{max} = \beta \cdot |A|$)")
    ax1.set_ylabel(r"$\bar Q$ - середнє значення цільової функції")
    ax1.set_title("Залежність якості розв'язку від β")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xscale("log")
    ax1.set_xticks(betas)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))

    ax2.errorbar(betas, t_means, yerr=t_stds,
                 marker="s", color="#c0392b", linewidth=2, capsize=4)
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel(r"$\bar t$, с - середній час роботи")
    ax2.set_title("Залежність часу роботи від β")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xscale("log")
    ax2.set_xticks(betas)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))

    if problem_info:
        fig.suptitle(f"Експеримент 3.3.2: визначення β.   {problem_info}",
                     fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(filepath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_rho_experiment(rhos, Q_G_means, Q_LS_means,
                        delta_means, delta_stds,
                        filepath: str,
                        problem_info: str = ""):
    """Графіки Q̄_G(ρ_F), Q̄_LS(ρ_F), δ̄(ρ_F) для експерименту 3.3.3."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(rhos, Q_G_means, marker="o", color="#27ae60",
             label="Жадібний", linewidth=2)
    ax1.plot(rhos, Q_LS_means, marker="s", color="#2980b9",
             label="Локальний пошук", linewidth=2)
    ax1.set_xlabel(r"$\rho_F$ - частка заборонених клітин")
    ax1.set_ylabel(r"$\bar Q$ - середнє значення цільової функції")
    ax1.set_title("Якість розв'язків від ρ_F")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(rhos, [100 * d for d in delta_means],
                 yerr=[100 * s for s in delta_stds],
                 marker="^", color="#8e44ad", linewidth=2, capsize=4)
    ax2.set_xlabel(r"$\rho_F$ - частка заборонених клітин")
    ax2.set_ylabel(r"$\bar\delta$, % - відносне покращення локальним пошуком")
    ax2.set_title("Відносне покращення локального пошуку")
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax2.grid(True, alpha=0.3)

    if problem_info:
        fig.suptitle(f"Експеримент 3.3.3: вплив ρ_F.   {problem_info}",
                     fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(filepath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_budget_experiment(alphas, Q_G_means, Q_LS_means,
                           delta_means, delta_stds,
                           filepath: str,
                           problem_info: str = ""):
    """Графіки Q̄(α_B) та δ̄(α_B) для експерименту з бюджетом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(alphas, Q_G_means, marker="o", color="#27ae60",
             label="Жадібний", linewidth=2)
    ax1.plot(alphas, Q_LS_means, marker="s", color="#2980b9",
             label="Локальний пошук", linewidth=2)
    ax1.set_xlabel(r"$\alpha_B$ — коефіцієнт бюджету")
    ax1.set_ylabel(r"$\bar Q$ — середнє значення цільової функції")
    ax1.set_title("Залежність якості розв'язку від бюджету")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(alphas, [100 * d for d in delta_means],
                 yerr=[100 * s for s in delta_stds],
                 marker="^", color="#8e44ad", linewidth=2, capsize=4)
    ax2.set_xlabel(r"$\alpha_B$")
    ax2.set_ylabel(r"$\bar\delta$, % — відносне покращення ЛП над жадібним")
    ax2.set_title("Відносне покращення локального пошуку")
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax2.grid(True, alpha=0.3)

    if problem_info:
        fig.suptitle(f"Вплив бюджету (α_B).   {problem_info}",
                     fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(filepath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_structure_experiment(s_fs, s_names, Q_G_means, Q_LS_means,
                              delta_means, delta_stds,
                              filepath: str,
                              problem_info: str = ""):
    """Стовпчасті діаграми Q і δ для різних типів структур заборонених зон."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = list(range(len(s_fs)))
    width = 0.35

    ax1.bar([i - width / 2 for i in x], Q_G_means, width,
            color="#27ae60", label="Жадібний")
    ax1.bar([i + width / 2 for i in x], Q_LS_means, width,
            color="#2980b9", label="Локальний пошук")
    ax1.set_xlabel("Тип структури заборонених зон")
    ax1.set_ylabel(r"$\bar Q$ — середнє значення цільової функції")
    ax1.set_title("Якість розв'язків за типом структури F")
    ax1.set_xticks(x)
    ax1.set_xticklabels(s_names, fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, [100 * d for d in delta_means], color="#8e44ad", alpha=0.8)
    ax2.set_xlabel("Тип структури заборонених зон")
    ax2.set_ylabel(r"$\bar\delta$, %")
    ax2.set_title("Відносне покращення від жадібного до ЛП")
    ax2.set_xticks(x)
    ax2.set_xticklabels(s_names, fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax2.grid(True, alpha=0.3, axis="y")

    if problem_info:
        fig.suptitle(f"Вплив структури F.   {problem_info}",
                     fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(filepath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_size_experiment(sizes, Q_G_means, Q_LS_means,
                         t_G_means, t_LS_means,
                         delta_means, delta_stds,
                         filepath: str,
                         problem_info: str = ""):
    """Графіки Q̄(mn), t̄(mn), δ̄(mn) для експерименту 3.3.4."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(sizes, Q_G_means, marker="o", color="#27ae60",
            label="Жадібний", linewidth=2)
    ax.plot(sizes, Q_LS_means, marker="s", color="#2980b9",
            label="Локальний пошук", linewidth=2)
    ax.set_xlabel(r"$m \cdot n$ - розмір парку (клітин)")
    ax.set_ylabel(r"$\bar Q$")
    ax.set_title("Якість розв'язку від розмірності")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(sizes, t_G_means, marker="o", color="#27ae60",
            label="Жадібний", linewidth=2)
    ax.plot(sizes, t_LS_means, marker="s", color="#2980b9",
            label="Локальний пошук", linewidth=2)
    ax.set_xlabel(r"$m \cdot n$ - розмір парку (клітин)")
    ax.set_ylabel(r"$\bar t$, с")
    ax.set_title("Час роботи від розмірності")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[2]
    ax.errorbar(sizes, [100 * d for d in delta_means],
                yerr=[100 * s for s in delta_stds],
                marker="^", color="#8e44ad", linewidth=2, capsize=4)
    ax.set_xlabel(r"$m \cdot n$")
    ax.set_ylabel(r"$\bar\delta$, %")
    ax.set_title("Відносне покращення локальним пошуком")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.grid(True, alpha=0.3)

    if problem_info:
        fig.suptitle(f"Експеримент 3.3.4: вплив розмірності.   {problem_info}",
                     fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(filepath, dpi=140, bbox_inches="tight")
    plt.close(fig)
