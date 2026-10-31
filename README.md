# Optimisasi Lokasi Kopdes Merah Putih — Simulated Annealing

Final Project Konsep Kecerdasan Artifisial. Memilih *k* lokasi Kopdes dari kandidat
agar jangkauan, pemerataan, potensi ekonomi, dan aksesibilitas maksimal.

## Menjalankan
```
pip install numpy matplotlib pandas
python main.py demo                      # 1 skenario: tabel, peta sebelum/sesudah, konvergensi
python main.py demo --k 15 --radius 1.5  # ubah parameter
python main.py experiments --seeds 5     # semua skenario uji + studi parameter + skalabilitas
```
Hasil (PNG/CSV) disimpan di folder `results/`.

## Struktur
| File | Isi |
| --- | --- |
| `data.py` | Generator data sintetis (mirip kota) dan `load_csv()` untuk data nyata |
| `model.py` | State, constraint, fungsi objektif, laporan metrik |
| `optimizers.py` | Simulated Annealing, Hill Climbing (random restart), Random Search |
| `visualize.py` | Peta (map coloring per Kopdes, area belum terjangkau), konvergensi, boxplot |
| `experiments.py` | Skenario uji, studi parameter SA, studi skalabilitas |
| `main.py` | CLI |

## Formulasi
- **State**: array berisi *k* indeks kandidat.
- **Neighbor**: ganti satu Kopdes (60% ke kandidat terdekat, 40% ke kandidat acak).
- **Skor** (maksimalkan, komponen 0..1):
  `0.45·Jangkauan + 0.25·Pemerataan + 0.15·Ekonomi + 0.15·Akses`
  - Jangkauan = fraksi penduduk dalam radius R dari Kopdes terdekat (sudah memuat kepadatan)
  - Pemerataan = 1 − rata-rata jarak terbobot ke Kopdes terdekat (dipotong 3R) / 3R
  - Ekonomi = rata-rata skor ekonomi lokasi terpilih; Akses = 1 − rata-rata jarak ke jalan utama (ternormalisasi)
- **Hard constraint**: lahan ≥ 1.000 m² (kandidat disaring di awal); jarak antar Kopdes ≥ `min_sep` (default 1 km).
- **SA**: pendinginan geometrik `T ← α·T`, T0=0.05, α=0.995, 20 iterasi per suhu, berhenti di T=1e-4.

## Kompleksitas
- Satu evaluasi skor: O(k·P) (min jarak ke *k* Kopdes untuk P titik penduduk).
- SA: O(I·k·P) dengan I = (jumlah suhu × iterasi per suhu), jumlah suhu = ln(T_min/T0)/ln(α).
- Memori: O(N·P) untuk matriks jarak kandidat–penduduk (+ O(N²) kandidat–kandidat).
  Ini didukung hasil `results/skalabilitas.csv` (memori naik linear terhadap P dan N).

## Data nyata (Surabaya)
Siapkan dua CSV lalu jalankan `python main.py demo --pop population.csv --cand candidates.csv`:
- `population.csv`: `lat,lon,population` (mis. centroid kelurahan + jumlah penduduk BPS)
- `candidates.csv`: `id,lat,lon,land_area_m2,econ_score,road_dist_km`
  (`econ_score` 0..1, mis. proxy jumlah UMKM/pasar/POI; `road_dist_km` bisa dihitung dari OSM)
