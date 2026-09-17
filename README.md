# CASMI 2026 — Molecule ID from Mass Spectra

Entry for the [Enveda CASMI 2026 competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/)
($50,000 prize pool, closes 2026-12-14). Task: rank up to 25 candidate SMILES
per molecule, aggregating all spectra that belong to each molecule.

Kaggle notebook: [CASMI26 Baseline Predictor](https://www.kaggle.com/code/tyreejones393/casmi26-baseline-predictor/edit)

## Status

- **Rung 1 — sealed.** Runtime prediction pipeline: reads `test.parquet` at
  scoring time (works on visible and hidden test sets), searches a spectral
  index, writes `submission.csv`. TYREE's first scored CASMI submission:
  displayed public score **0.000** (2026-09-17) — that is the verified
  public number; it does not by itself prove no ranked candidate matched.
  The pipeline runs clean; the 0.000 is the starting line, not the verdict.
- **Rung 2 — measured 2026-09-17.** Exact-mass chemistry baseline: 20 ppm
  greedy fragment + neutral-loss matching, known-structure holdout top-1
  **0.947** (`src/exact_mass_search.py`, `src/benchmark_exact_mass.py`).
  Phi-frame, measured without any Cartesian grid on the phi maps
  (`src/benchmark_phi_frame_v3.py`, `models/phi_frame_v3_report.txt`):
  map2 phi-rung address match top-1 **0.940** (ranking rung — fixed ppm in
  m/z is a fixed rung tolerance, the coordinate absorbs the scale);
  map3 conjugate-interval multiset top-1 **0.723**; map4 reflection dropped
  (fragmentation has no reflection symmetry — domain fact, not a residue).
  3-witness lock: LOCKED 199/300 (66.3%) at **0.990**, MARGINAL 80/300 at
  0.787, REJECTED 21/300 (lock refuses; these are the genuinely hard
  spectra). Deployable molecule-balanced raw-peak index: 539,120 spectra /
  183,192 molecules (`models/casmi_raw_index.npz`, 55.7 MB).

## Pipeline

1. `src/build_index.py` — streams `train.parquet`, vectorizes Enveda-library
   spectra (1 Da bins, 1500 dims, intensity floor 0.01, sqrt transform,
   L2-normalized), caps at 150,000 vectors via memory-mapped arrays, sorts
   by precursor m/z, packs to `models/casmi_index.npz` (~11 MB).
2. `src/notebook_predict_v2.py` — the scored code. Loads the index from the
   attached Kaggle dataset, reads the competition's `test.parquet` at run
   time, groups spectra by molecule, cosine-searches a ±0.5 Da precursor
   window, aggregates the top-20 hits per spectrum (threshold 0.2), emits
   25 candidates per molecule, writes LF-only `submission.csv`.
3. The Kaggle notebook loader cell finds both files under `/kaggle/input`
   with glob — no internet, no hard-coded paths (scored reruns have
   internet off).

## Key lesson

Static CSV submissions can never score here: Kaggle replaces the visible
test set with hidden molecule IDs at scoring time. The notebook must read
the test data at run time. Four failed static attempts proved it; the
runtime predictor was the fix.

## Reproduce

```bash
pip install numpy pandas pyarrow
python src/build_index.py          # needs data/train.parquet
python src/notebook_predict_v2.py  # Kaggle paths; adapt INPUT_DIR locally
```

## License

See [LICENSING.md](LICENSING.md). All free, all phi — free non-commercial
use with attribution to Tyree Jones (tyofthestarz).
