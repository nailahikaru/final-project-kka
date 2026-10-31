"""Visualisasi peta hasil optimisasi (matplotlib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def draw_solution(ax, prob, sol, title):
    """Peta: titik penduduk diwarnai menurut Kopdes terdekat (map coloring);
    penduduk di luar radius layanan ditandai merah (belum terjangkau)."""
    ds = prob.ds
    d = prob.D[sol]                       # (k, P)
    nearest = d.argmin(axis=0)
    covered = d.min(axis=0) <= prob.R
    cmap = plt.get_cmap("tab20")
    size = 4 + 40 * ds.pop_w / ds.pop_w.max()

    for line in ds.roads:
        ax.plot(line[:, 0], line[:, 1], color="#999", lw=2.5, alpha=0.6, zorder=1)
    ax.scatter(*ds.pop_xy[~covered].T, s=size[~covered], c="#d62728", marker="x",
               alpha=0.6, linewidths=0.8, label="Belum terjangkau", zorder=2)
    ax.scatter(*ds.pop_xy[covered].T, s=size[covered],
               c=[cmap(i % 20) for i in nearest[covered]], alpha=0.65, zorder=2)
    ax.scatter(*prob.xy.T, s=6, c="#bbb", zorder=2, label="Kandidat layak")
    for c in sol:
        ax.add_patch(Circle(prob.xy[c], prob.R, fill=False, ec="k", lw=0.7,
                            ls="--", alpha=0.5, zorder=3))
    ax.scatter(*prob.xy[sol].T, marker="*", s=220, c="gold", ec="k", zorder=4,
               label="Kopdes terpilih")
    rep = prob.report(sol)
    ax.set_title(f"{title}\nterlayani {rep['pop_served_pct']:.1f}% | "
                 f"skor {prob.score(sol):.4f}", fontsize=10)
    x0, x1, y0, y1 = ds.bounds
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")


def plot_before_after(prob, result, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    draw_solution(axes[0], prob, result.init_sol, "SEBELUM optimisasi (acak)")
    draw_solution(axes[1], prob, result.best_sol, f"SESUDAH ({result.name})")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_convergence(results, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        ax.plot(r.history, label=f"{r.name} ({r.best_score:.4f})")
    ax.set_xlabel("Jumlah evaluasi fungsi objektif")
    ax.set_ylabel("Skor terbaik sejauh ini")
    ax.set_title("Konvergensi")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_boxplot(scores_by_algo, path, title="Sebaran skor akhir"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(list(scores_by_algo.values()), tick_labels=list(scores_by_algo.keys()))
    ax.set_ylabel("Skor akhir")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
