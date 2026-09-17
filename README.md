# CASMI 2026 — Molecule ID from Mass Spectra

Entry for the [Enveda CASMI 2026 competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/)
($50,000 prize pool, closes 2026-12-14). Task: rank up to 25 candidate SMILES
per molecule, aggregating all spectra that belong to each molecule.

Kaggle notebook: [CASMI26 Baseline Predictor](https://www.kaggle.com/code/tyreejones393/casmi26-baseline-predictor/edit)

## Status

- **Rung 1 — sealed.** Runtime prediction pipeline: reads `test.parquet` at
  scoring time (works on visible and hidden test sets), searches a spectral
  index, writes `submission.csv`. First scored submission in history:
  public score **0.000** (2026-09-17). The pipeline runs clean; the ranking
  hit nothing — the number is the starting line, not the verdict.
- **Rung 2 — in progress.** Diagnosis: coverage (true molecules missing from
  the index) vs method (similarity too weak). Upgrades on the bench:
  exact-mass (ppm) fragment matching — the data is high-resolution
  (4 decimals) and 1-Da binning throws it away; adduct → neutral mass →
  formula enumeration; neutral-loss fingerprints.

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
