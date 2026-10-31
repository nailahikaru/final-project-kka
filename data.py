"""Pembuatan / pemuatan data untuk optimisasi lokasi Kopdes Merah Putih.

Semua koordinat internal memakai satuan kilometer (bidang datar) supaya
perhitungan jarak cukup dengan jarak Euclidean. Untuk data nyata (lat/lon),
gunakan load_csv() yang memproyeksikan koordinat ke kilometer.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dataset:
    pop_xy: np.ndarray        # (P, 2) koordinat titik penduduk (km)
    pop_w: np.ndarray         # (P,)  jumlah penduduk di titik tsb
    cand_xy: np.ndarray       # (N, 2) koordinat kandidat lokasi (km)
    cand_land: np.ndarray     # (N,)  luas lahan tersedia (m2)
    cand_econ: np.ndarray     # (N,)  skor potensi ekonomi (0..1)
    cand_road: np.ndarray     # (N,)  jarak ke jalan utama (km)
    roads: list = field(default_factory=list)  # polyline jalan utama (untuk peta)
    bounds: tuple = (0.0, 18.0, 0.0, 22.0)     # xmin, xmax, ymin, ymax


# ----------------------------------------------------------------------------
# Data sintetis (mirip bentuk kota: beberapa pusat keramaian + jalan utama)
# ----------------------------------------------------------------------------
def _dist_point_polyline(pts, line):
    """Jarak minimum tiap titik ke sebuah polyline."""
    best = np.full(len(pts), np.inf)
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        t = np.clip(((pts - a) @ ab) / (ab @ ab), 0.0, 1.0)
        proj = a + t[:, None] * ab
        best = np.minimum(best, np.linalg.norm(pts - proj, axis=1))
    return best


def make_synthetic(n_pop=2000, n_cand=300, total_pop=2_900_000, seed=0,
                   width=18.0, height=22.0):
    rng = np.random.default_rng(seed)

    # pusat-pusat permukiman (x, y, bobot, sebaran)
    centers = np.array([
        [8, 11, 3.0, 2.2], [5, 15, 2.0, 1.8], [12, 8, 2.0, 2.0],
        [9, 4, 1.5, 1.8], [14, 15, 1.5, 1.6], [4, 6, 1.2, 1.5],
        [11, 19, 1.0, 1.4], [15, 11, 1.0, 1.5],
    ])
    pick = rng.choice(len(centers), size=n_pop, p=centers[:, 2] / centers[:, 2].sum())
    pop_xy = centers[pick, :2] + rng.normal(size=(n_pop, 2)) * centers[pick, 3:4]
    pop_xy[:, 0] = np.clip(pop_xy[:, 0], 0.2, width - 0.2)
    pop_xy[:, 1] = np.clip(pop_xy[:, 1], 0.2, height - 0.2)
    w = rng.lognormal(mean=0.0, sigma=0.6, size=n_pop)
    pop_w = w / w.sum() * total_pop

    # jalan utama (polyline)
    roads = [
        np.array([[1, 1], [6, 8], [9, 12], [10, 21]], float),     # utara-selatan
        np.array([[0.5, 14], [6, 12], [12, 9], [17.5, 8]], float),  # barat-timur
        np.array([[2, 20], [7, 16], [13, 14], [17.5, 17]], float),  # diagonal utara
        np.array([[9, 1], [11, 6], [13, 10]], float),               # akses selatan
    ]

    # kandidat: dekat permukiman (dipilih sesuai kepadatan) + sedikit acak
    idx = rng.choice(n_pop, size=n_cand, p=pop_w / pop_w.sum())
    cand_xy = pop_xy[idx] + rng.normal(scale=0.6, size=(n_cand, 2))
    cand_xy[:, 0] = np.clip(cand_xy[:, 0], 0.2, width - 0.2)
    cand_xy[:, 1] = np.clip(cand_xy[:, 1], 0.2, height - 0.2)

    # luas lahan (m2): sekitar 30% kandidat < 1.000 m2 -> terfilter constraint
    cand_land = rng.lognormal(mean=np.log(1500), sigma=0.7, size=n_cand)

    # potensi ekonomi: proxy kepadatan penduduk sekitar + noise (0..1)
    d = np.linalg.norm(cand_xy[:, None, :] - pop_xy[None, :, :], axis=2)
    density = (pop_w[None, :] * np.exp(-(d / 1.5) ** 2)).sum(axis=1)
    econ = density / density.max() * 0.7 + rng.random(n_cand) * 0.3
    cand_econ = (econ - econ.min()) / (econ.max() - econ.min())

    cand_road = np.min([_dist_point_polyline(cand_xy, r) for r in roads], axis=0)

    return Dataset(pop_xy, pop_w, cand_xy, cand_land, cand_econ, cand_road,
                   roads, (0.0, width, 0.0, height))


# ----------------------------------------------------------------------------
# Data nyata dari CSV (lat/lon)
# ----------------------------------------------------------------------------
def load_csv(pop_path, cand_path):
    """Muat data nyata.

    population.csv : lat, lon, population
    candidates.csv : id, lat, lon, land_area_m2, econ_score, road_dist_km
    econ_score = 0..1 (mis. proxy jumlah UMKM/pasar/POI yang dinormalisasi).
    """
    import pandas as pd

    pop = pd.read_csv(pop_path)
    cand = pd.read_csv(cand_path)
    lat0 = np.mean(np.r_[pop["lat"], cand["lat"]])
    lon0 = np.mean(np.r_[pop["lon"], cand["lon"]])

    def proj(lat, lon):  # proyeksi ekuirektangular -> km
        x = (np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0))
        y = (np.asarray(lat) - lat0) * 110.57
        return np.c_[x, y]

    pop_xy = proj(pop["lat"], pop["lon"])
    cand_xy = proj(cand["lat"], cand["lon"])
    allp = np.vstack([pop_xy, cand_xy])
    bounds = (allp[:, 0].min() - 1, allp[:, 0].max() + 1,
              allp[:, 1].min() - 1, allp[:, 1].max() + 1)
    return Dataset(pop_xy, pop["population"].to_numpy(float), cand_xy,
                   cand["land_area_m2"].to_numpy(float),
                   cand["econ_score"].to_numpy(float),
                   cand["road_dist_km"].to_numpy(float), [], bounds)
