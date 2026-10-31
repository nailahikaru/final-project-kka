"""Skenario uji: perbandingan algoritma, variasi parameter, dan skalabilitas."""
import time
import tracemalloc

import numpy as np
import pandas as pd

from data import make_synthetic
from model import Problem, Weights
from optimizers import hill_climbing, random_search, simulated_annealing


def run_all(prob, seed, sa_kwargs=None):
    """Jalankan SA, HC, RS dengan anggaran evaluasi yang sama."""
    sa_kwargs = sa_kwargs or {}
    sa = simulated_annealing(prob, np.random.default_rng(seed), **sa_kwargs)
    budget = sa.evals
    hc = hill_climbing(prob, np.random.default_rng(seed), max_evals=budget,
                       patience=1000)
    rs = random_search(prob, np.random.default_rng(seed), max_evals=budget)
    return [sa, hc, rs]


def _row(scn, prob, res, seed):
    rep = prob.report(res.best_sol)
    return dict(skenario=scn, algoritma=res.name, seed=seed,
                skor=res.best_score, skor_awal=res.init_score,
                terlayani_pct=rep["pop_served_pct"],
                jarak_rata2_km=rep["mean_dist"], evaluasi=res.evals,
                waktu_s=res.seconds)


def scenario_grid(n_seeds=5):
    """Variasi k, radius layanan, bobot, dan ukuran data."""
    ds_cache = {}
    def get_ds(n_pop, n_cand):
        if (n_pop, n_cand) not in ds_cache:
            ds_cache[(n_pop, n_cand)] = make_synthetic(n_pop, n_cand, seed=0)
        return ds_cache[(n_pop, n_cand)]

    scenarios = []
    for k in (5, 10, 20):
        scenarios.append((f"k={k}", dict(k=k), (2000, 300)))
    for R in (1.0, 2.0, 3.0):
        scenarios.append((f"R={R}km", dict(k=10, radius=R), (2000, 300)))
    scenarios.append(("bobot: fokus jangkauan",
                      dict(k=10, weights=Weights(0.7, 0.1, 0.1, 0.1)), (2000, 300)))
    scenarios.append(("bobot: fokus ekonomi+akses",
                      dict(k=10, weights=Weights(0.25, 0.1, 0.35, 0.3)), (2000, 300)))
    scenarios.append(("data kecil (500 pend, 100 kand)", dict(k=10), (500, 100)))
    scenarios.append(("data besar (5000 pend, 600 kand)", dict(k=10), (5000, 600)))

    rows = []
    for name, kw, (n_pop, n_cand) in scenarios:
        prob = Problem(get_ds(n_pop, n_cand), **kw)
        for seed in range(n_seeds):
            for res in run_all(prob, seed):
                rows.append(_row(name, prob, res, seed))
        print(f"  selesai: {name}")
    return pd.DataFrame(rows)


def sa_parameter_study(n_seeds=5):
    """Pengaruh cooling rate & suhu awal terhadap kualitas dan waktu."""
    prob = Problem(make_synthetic(seed=0), k=10)
    rows = []
    for alpha in (0.95, 0.99, 0.995, 0.999):
        for T0 in (0.01, 0.05, 0.2):
            for seed in range(n_seeds):
                r = simulated_annealing(prob, np.random.default_rng(seed),
                                        T0=T0, alpha=alpha)
                rows.append(dict(alpha=alpha, T0=T0, seed=seed, skor=r.best_score,
                                 evaluasi=r.evals, waktu_s=r.seconds))
    return pd.DataFrame(rows)


def scaling_study(seed=0):
    """Waktu & memori SA terhadap jumlah titik penduduk P dan jumlah Kopdes k."""
    rows = []
    sa_kw = dict(alpha=0.99)  # anggaran tetap ~ 1.2k temperatur x 20 iterasi
    for n_pop in (500, 1000, 2000, 4000, 8000):
        ds = make_synthetic(n_pop, 300, seed=0)
        tracemalloc.start()
        prob = Problem(ds, k=10)
        r = simulated_annealing(prob, np.random.default_rng(seed), **sa_kw)
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        rows.append(dict(variabel="P (titik penduduk)", nilai=n_pop,
                         evaluasi=r.evals, waktu_s=r.seconds,
                         waktu_per_eval_ms=1000 * r.seconds / r.evals, memori_MB=peak))
    ds = make_synthetic(2000, 300, seed=0)
    for k in (5, 10, 20, 40):
        tracemalloc.start()
        prob = Problem(ds, k=k)
        r = simulated_annealing(prob, np.random.default_rng(seed), **sa_kw)
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        rows.append(dict(variabel="k (jumlah Kopdes)", nilai=k,
                         evaluasi=r.evals, waktu_s=r.seconds,
                         waktu_per_eval_ms=1000 * r.seconds / r.evals, memori_MB=peak))
    for n_cand in (100, 200, 400, 800):
        ds = make_synthetic(2000, n_cand, seed=0)
        tracemalloc.start()
        prob = Problem(ds, k=10)
        r = simulated_annealing(prob, np.random.default_rng(seed), **sa_kw)
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        rows.append(dict(variabel="N (kandidat)", nilai=n_cand,
                         evaluasi=r.evals, waktu_s=r.seconds,
                         waktu_per_eval_ms=1000 * r.seconds / r.evals, memori_MB=peak))
    return pd.DataFrame(rows)
