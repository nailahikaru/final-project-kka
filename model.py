"""Formulasi masalah: state, constraint, dan fungsi objektif.

State   : array berisi k indeks kandidat (lokasi Kopdes yang dipilih).
Objektif: maksimalkan skor gabungan (semua komponen dinormalisasi 0..1)

    skor = w_cov*Jangkauan + w_eq*Pemerataan + w_econ*Ekonomi + w_acc*Akses

 - Jangkauan  : fraksi penduduk yang berada dalam radius layanan R
                (sudah memuat unsur kepadatan penduduk, karena memakai bobot
                 jumlah penduduk tiap titik)
 - Pemerataan : 1 - rata-rata jarak (terbobot, dipotong 3R) penduduk ke Kopdes
                terdekat / 3R
 - Ekonomi    : rata-rata skor potensi ekonomi lokasi terpilih
 - Akses      : 1 - rata-rata jarak lokasi terpilih ke jalan utama (ternormalisasi)

Hard constraint:
 - luas lahan >= min_land (1.000 m2)  -> kandidat tak layak dibuang di awal
 - jarak antar Kopdes >= min_sep      -> dijaga di fungsi init & neighbor
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Weights:
    cov: float = 0.45
    eq: float = 0.25
    econ: float = 0.15
    acc: float = 0.15

    def normalized(self):
        s = self.cov + self.eq + self.econ + self.acc
        return Weights(self.cov / s, self.eq / s, self.econ / s, self.acc / s)


class Problem:
    def __init__(self, ds, k=10, radius=2.0, weights=None, min_sep=1.0,
                 min_land=1000.0, n_neighbors=15):
        self.ds = ds
        self.k = k
        self.R = radius
        self.w = (weights or Weights()).normalized()
        self.min_sep = min_sep
        self.min_land = min_land

        # ---- hard constraint lahan: saring kandidat --------------------------
        self.cand_ids = np.where(ds.cand_land >= min_land)[0]   # indeks ke Dataset
        if len(self.cand_ids) < k:
            raise ValueError("Kandidat layak kurang dari k")
        self.xy = ds.cand_xy[self.cand_ids]
        econ = ds.cand_econ[self.cand_ids]
        road = ds.cand_road[self.cand_ids]
        self.econ = (econ - econ.min()) / (np.ptp(econ) + 1e-12)
        self.road = road / (road.max() + 1e-12)
        self.M = len(self.cand_ids)

        # ---- pra-hitung jarak (dipakai berulang oleh fungsi objektif) --------
        self.pop_w = ds.pop_w / ds.pop_w.sum()           # bobot penduduk (jumlah=1)
        self.D = self._cdist(self.xy, ds.pop_xy)         # (M, P) kandidat->penduduk
        self.CC = self._cdist(self.xy, self.xy)          # (M, M) kandidat->kandidat
        self.nn = np.argsort(self.CC, axis=1)[:, 1:n_neighbors + 1]  # tetangga dekat
        self.evals = 0

    @staticmethod
    def _cdist(a, b):
        d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
        return np.sqrt(np.maximum(d2, 0.0))

    # ---------------------------------------------------------------- objektif
    def components(self, sol):
        dmin = self.D[sol].min(axis=0)                     # jarak ke Kopdes terdekat
        cov = float(self.pop_w[dmin <= self.R].sum())
        cap = 3 * self.R
        eq = 1.0 - float((self.pop_w * np.minimum(dmin, cap)).sum() / cap)
        econ = float(self.econ[sol].mean())
        acc = 1.0 - float(self.road[sol].mean())
        return dict(coverage=cov, equity=eq, econ=econ, access=acc,
                    mean_dist=float((self.pop_w * dmin).sum()))

    def score(self, sol):
        """Skor yang dimaksimalkan (menambah hitungan evaluasi)."""
        self.evals += 1
        dmin = self.D[sol].min(axis=0)
        cov = self.pop_w[dmin <= self.R].sum()
        cap = 3 * self.R
        eq = 1.0 - (self.pop_w * np.minimum(dmin, cap)).sum() / cap
        w = self.w
        return float(w.cov * cov + w.eq * eq + w.econ * self.econ[sol].mean()
                     + w.acc * (1.0 - self.road[sol].mean()))

    # --------------------------------------------------------- state & neighbor
    def random_solution(self, rng):
        """Solusi acak yang memenuhi jarak minimal antar Kopdes."""
        for _ in range(200):
            order = rng.permutation(self.M)
            sol = []
            for c in order:
                if all(self.CC[c, s] >= self.min_sep for s in sol):
                    sol.append(c)
                    if len(sol) == self.k:
                        return np.array(sol)
        raise RuntimeError("Gagal membuat solusi awal: kurangi k atau min_sep")

    def neighbor(self, sol, rng):
        """Ganti satu Kopdes dengan kandidat lain (60% geser lokal, 40% lompat)."""
        for _ in range(50):
            i = rng.integers(self.k)
            old = sol[i]
            c = rng.choice(self.nn[old]) if rng.random() < 0.6 else rng.integers(self.M)
            if (sol == c).any():
                continue
            others = np.delete(sol, i)
            if (self.CC[c, others] >= self.min_sep).all():
                new = sol.copy()
                new[i] = c
                return new
        return None

    # --------------------------------------------------------------- pelaporan
    def report(self, sol):
        comp = self.components(sol)
        dmin = self.D[sol].min(axis=0)
        uncovered = dmin > self.R
        total_pop = self.ds.pop_w.sum()
        comp["pop_served_pct"] = 100 * comp["coverage"]
        comp["pop_unserved"] = float(self.ds.pop_w[uncovered].sum())
        comp["total_pop"] = float(total_pop)
        comp["n_unserved_points"] = int(uncovered.sum())
        return comp
