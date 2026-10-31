"""Algoritma pencarian: Simulated Annealing (utama), Hill Climbing with random
restart, dan Random Search (baseline). Semuanya MEMAKSIMALKAN problem.score().

Setiap algoritma mengembalikan Result; `history` berisi skor terbaik sampai saat
itu pada setiap evaluasi (dipakai untuk grafik konvergensi).
"""
import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    name: str
    best_sol: np.ndarray
    best_score: float
    init_sol: np.ndarray
    init_score: float
    history: np.ndarray
    evals: int
    seconds: float


def simulated_annealing(prob, rng, T0=0.05, T_min=1e-4, alpha=0.995,
                        iters_per_T=20, name="Simulated Annealing"):
    """Pendinginan geometrik: T <- alpha * T. Solusi lebih buruk diterima dengan
    peluang exp(delta/T) sehingga bisa keluar dari local optimum."""
    t0 = time.perf_counter()
    prob.evals = 0
    cur = prob.random_solution(rng)
    cur_s = prob.score(cur)
    init_sol, init_s = cur.copy(), cur_s
    best, best_s = cur.copy(), cur_s
    hist = [best_s]

    T = T0
    while T > T_min:
        for _ in range(iters_per_T):
            cand = prob.neighbor(cur, rng)
            if cand is None:
                continue
            s = prob.score(cand)
            delta = s - cur_s
            if delta > 0 or rng.random() < math.exp(delta / T):
                cur, cur_s = cand, s
                if cur_s > best_s:
                    best, best_s = cur.copy(), cur_s
            hist.append(best_s)
        T *= alpha
    return Result(name, best, best_s, init_sol, init_s, np.array(hist),
                  prob.evals, time.perf_counter() - t0)


def hill_climbing(prob, rng, max_evals=25000, restarts=5, patience=300,
                  name="Hill Climbing (restart)"):
    """Hanya menerima tetangga yang lebih baik; berhenti jika `patience`
    percobaan berturut-turut tidak membaik, lalu restart dari solusi acak."""
    t0 = time.perf_counter()
    prob.evals = 0
    per_restart = max_evals // restarts
    best = None
    best_s = -np.inf
    init_sol = init_s = None
    hist = []
    for r in range(restarts):
        cur = prob.random_solution(rng)
        cur_s = prob.score(cur)
        if r == 0:
            init_sol, init_s = cur.copy(), cur_s
        if cur_s > best_s:
            best, best_s = cur.copy(), cur_s
        hist.append(best_s)
        used, fails = 1, 0
        while used < per_restart and fails < patience:
            cand = prob.neighbor(cur, rng)
            if cand is None:
                break
            s = prob.score(cand)
            used += 1
            if s > cur_s:
                cur, cur_s, fails = cand, s, 0
                if s > best_s:
                    best, best_s = cand.copy(), s
            else:
                fails += 1
            hist.append(best_s)
    return Result(name, best, best_s, init_sol, init_s, np.array(hist),
                  prob.evals, time.perf_counter() - t0)


def random_search(prob, rng, max_evals=25000, name="Random Search"):
    """Baseline: ambil solusi acak sebanyak max_evals, simpan yang terbaik."""
    t0 = time.perf_counter()
    prob.evals = 0
    best = None
    best_s = -np.inf
    hist = []
    init_sol = init_s = None
    for i in range(max_evals):
        sol = prob.random_solution(rng)
        s = prob.score(sol)
        if i == 0:
            init_sol, init_s = sol.copy(), s
        if s > best_s:
            best, best_s = sol, s
        hist.append(best_s)
    return Result(name, best, best_s, init_sol, init_s, np.array(hist),
                  prob.evals, time.perf_counter() - t0)
