"""Optimisasi lokasi Kopdes Merah Putih dengan Simulated Annealing.

Contoh:
  python main.py demo                       # satu skenario + peta + konvergensi
  python main.py demo --k 15 --radius 1.5
  python main.py experiments                # semua skenario uji (lebih lama)
  python main.py demo --pop population.csv --cand candidates.csv   # data nyata
"""
import argparse
import os

import numpy as np
import pandas as pd

from data import load_csv, make_synthetic
from experiments import (run_all, scaling_study, scenario_grid,
                         sa_parameter_study)
from model import Problem, Weights
from visualize import plot_before_after, plot_boxplot, plot_convergence

OUT = "results"


def cmd_demo(a):
    os.makedirs(OUT, exist_ok=True)
    ds = load_csv(a.pop, a.cand) if a.pop else make_synthetic(seed=a.seed)
    w = Weights(a.w_cov, a.w_eq, a.w_econ, a.w_acc)
    prob = Problem(ds, k=a.k, radius=a.radius, weights=w, min_sep=a.min_sep)
    print(f"Kandidat total {len(ds.cand_land)} -> layak (lahan >= 1000 m2): {prob.M}")

    sa, hc, rs = run_all(prob, a.seed)
    print(f"\n{'Algoritma':26s}{'Skor':>9s}{'Terlayani':>11s}{'Jarak rata2':>13s}"
          f"{'Evaluasi':>10s}{'Waktu(s)':>10s}")
    print(f"{'Acak (sebelum optimisasi)':26s}{sa.init_score:9.4f}"
          f"{prob.report(sa.init_sol)['pop_served_pct']:10.1f}%"
          f"{prob.report(sa.init_sol)['mean_dist']:11.2f}km")
    for r in (rs, hc, sa):
        rep = prob.report(r.best_sol)
        print(f"{r.name:26s}{r.best_score:9.4f}{rep['pop_served_pct']:10.1f}%"
              f"{rep['mean_dist']:11.2f}km{r.evals:10d}{r.seconds:10.2f}")

    rep = prob.report(sa.best_sol)
    print(f"\nPenduduk belum terjangkau (SA): {rep['pop_unserved']:,.0f} jiwa "
          f"({100 - rep['pop_served_pct']:.1f}%)")
    print("Lokasi rekomendasi (id kandidat, x_km, y_km, lahan m2, ekonomi, jarak jalan km):")
    for c in sa.best_sol:
        i = prob.cand_ids[c]
        print(f"  #{i:4d}  ({ds.cand_xy[i,0]:6.2f}, {ds.cand_xy[i,1]:6.2f})  "
              f"{ds.cand_land[i]:7.0f}  {ds.cand_econ[i]:.2f}  {ds.cand_road[i]:.2f}")

    plot_before_after(prob, sa, f"{OUT}/peta_sebelum_sesudah.png")
    plot_convergence([sa, hc, rs], f"{OUT}/konvergensi.png")
    print(f"\nGambar tersimpan di {OUT}/")


def cmd_experiments(a):
    os.makedirs(OUT, exist_ok=True)
    print("1/3 Skenario perbandingan algoritma ...")
    df = scenario_grid(a.seeds)
    df.to_csv(f"{OUT}/hasil_skenario.csv", index=False)
    summ = (df.groupby(["skenario", "algoritma"])
              .agg(skor_mean=("skor", "mean"), skor_std=("skor", "std"),
                   terlayani_pct=("terlayani_pct", "mean"),
                   waktu_s=("waktu_s", "mean"), evaluasi=("evaluasi", "mean"))
              .round(4))
    summ.to_csv(f"{OUT}/ringkasan_skenario.csv")
    print(summ.to_string())
    base = df[df.skenario == "k=10"]
    plot_boxplot({n: g.skor.to_numpy() for n, g in base.groupby("algoritma")},
                 f"{OUT}/boxplot_k10.png", "Sebaran skor akhir (k=10, 5 seed)")

    print("\n2/3 Studi parameter SA ...")
    sp = sa_parameter_study(a.seeds)
    sp_sum = sp.groupby(["alpha", "T0"]).agg(skor_mean=("skor", "mean"),
                                             skor_std=("skor", "std"),
                                             evaluasi=("evaluasi", "mean"),
                                             waktu_s=("waktu_s", "mean")).round(4)
    sp_sum.to_csv(f"{OUT}/studi_parameter_sa.csv")
    print(sp_sum.to_string())

    print("\n3/3 Studi skalabilitas (waktu & memori) ...")
    sc = scaling_study().round(4)
    sc.to_csv(f"{OUT}/skalabilitas.csv", index=False)
    print(sc.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    d = sub.add_parser("demo")
    d.add_argument("--k", type=int, default=10)
    d.add_argument("--radius", type=float, default=2.0, help="radius layanan (km)")
    d.add_argument("--min-sep", type=float, default=1.0, help="jarak minimal antar Kopdes (km)")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--w-cov", type=float, default=0.45)
    d.add_argument("--w-eq", type=float, default=0.25)
    d.add_argument("--w-econ", type=float, default=0.15)
    d.add_argument("--w-acc", type=float, default=0.15)
    d.add_argument("--pop", help="CSV penduduk (lat,lon,population)")
    d.add_argument("--cand", help="CSV kandidat (id,lat,lon,land_area_m2,econ_score,road_dist_km)")
    d.set_defaults(fn=cmd_demo)
    e = sub.add_parser("experiments")
    e.add_argument("--seeds", type=int, default=5)
    e.set_defaults(fn=cmd_experiments)
    a = p.parse_args()
    if a.cmd is None:
        a = p.parse_args(["demo"])
    a.fn(a)


if __name__ == "__main__":
    main()
