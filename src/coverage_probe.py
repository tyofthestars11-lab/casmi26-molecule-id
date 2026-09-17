"""
Lightweight coverage probe — golden-ratio edition.

The full diagnostic got OOM-killed: it held the 900MB float32 matrix in RAM
while the benchmark ran beside it. This probe takes the 1-share: memory-mapped
float16 index (nothing loaded), per-spectrum window slices, incremental
writes. Peak RSS stays under 1/phi^2 of RAM.

Measures, for 250 molecules NOT in the 150k index (forced-zero design):
  - empty_windows: spectra with no index vector within +/-0.5 Da
  - max_sim distribution: best cosine similarity found per molecule
  - fallback_rate: molecules whose best sim < 0.2 (would hit closest-match fallback)
"""
import os
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict

ROOT = os.path.expanduser('~/workspace/kaggle-casmi26')
MODELS = os.path.join(ROOT, 'models')
N_BINS = 1500
PRECURSOR_TOL = 0.5
SIM_THRESHOLD = 0.2
OUT = os.path.join(MODELS, 'coverage_result.txt')


def spectrum_to_vector(mzs, intensities, precursor_mz):
    vec = np.zeros(N_BINS, dtype=np.float32)
    mzs = np.asarray(mzs)
    intensities = np.asarray(intensities)
    mask = (intensities >= 0.01) & (mzs <= precursor_mz + 2.0) & (mzs < N_BINS)
    bins = np.clip(mzs[mask].astype(int), 0, N_BINS - 1)
    for b, w in zip(bins, intensities[mask]):
        vec[b] += w
    vec = np.sqrt(vec)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


print("Mapping index (memmap, nothing loaded)...", flush=True)
precursors = np.load(os.path.join(MODELS, 'search_precursors.npy'))
vectors = np.memmap(os.path.join(MODELS, 'search_vectors_f16.npy'),
                    dtype=np.float16, mode='r',
                    shape=(len(precursors), N_BINS))
smiles_arr = np.load(os.path.join(MODELS, 'casmi_index.npz'))['smiles']
idx_smiles = set(str(s) for s in smiles_arr)
del smiles_arr
print(f"Index: {len(precursors)} vectors (memmap)", flush=True)

print("Collecting held-out molecules...", flush=True)
pf = pq.ParquetFile(os.path.join(ROOT, 'data/train.parquet'))
held = defaultdict(list)
MAX_MOLS = 250
bnum = 0
for batch in pf.iter_batches(batch_size=100000,
        columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz',
                 'normalized_smiles', 'ingest_lib']):
    bnum += 1
    if bnum <= 2:
        continue
    if len(held) >= MAX_MOLS:
        break
    df = batch.to_pandas()
    df = df[df['ingest_lib'].isin(['enveda-np-examples', 'enveda-180'])]
    for _, row in df.iterrows():
        smi = str(row['normalized_smiles'])
        if smi in idx_smiles:
            continue
        if smi not in held and len(held) >= MAX_MOLS:
            continue
        try:
            v = spectrum_to_vector(row['ms2_mzs'],
                                   row['ms2_normalized_intensities'],
                                   float(row['precursor_mz']))
        except Exception:
            continue
        if np.linalg.norm(v) == 0:
            continue
        held[smi].append((float(row['precursor_mz']), v))
print(f"Held-out molecules: {len(held)}", flush=True)

mols_empty = 0
mols_fallback = 0
best_sims = []
n = 0
with open(OUT, 'w') as f:
    for i, (smi, spectra) in enumerate(held.items()):
        mol_best = 0.0
        mol_empty = True
        for prec, qvec in spectra:
            lo = np.searchsorted(precursors, prec - PRECURSOR_TOL, side='left')
            hi = np.searchsorted(precursors, prec + PRECURSOR_TOL, side='right')
            if hi <= lo:
                continue
            mol_empty = False
            sims = np.asarray(vectors[lo:hi], dtype=np.float32) @ qvec
            mol_best = max(mol_best, float(sims.max()))
        n += 1
        best_sims.append(mol_best)
        if mol_empty:
            mols_empty += 1
        if mol_best < SIM_THRESHOLD:
            mols_fallback += 1
        f.write(f"{smi}\tbest_sim={mol_best:.4f}\tempty={mol_empty}\n")
        f.flush()
        if (i + 1) % 50 == 0:
            print(f"  probed {i+1}/{len(held)}", flush=True)

best_sims = np.array(best_sims)
result = (f"COVERAGE n={n} empty_window_rate={mols_empty/n:.3f} "
          f"fallback_rate={mols_fallback/n:.3f} "
          f"best_sim_median={np.median(best_sims):.3f} "
          f"best_sim_p90={np.percentile(best_sims, 90):.3f}")
print(result, flush=True)
with open(OUT, 'a') as f:
    f.write(result + "\n")
print(f"Wrote {OUT}", flush=True)
