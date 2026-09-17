"""
Coverage diagnostic, rerun — file-based so the RESULT always lands.

Same design as the first attempt: 250 molecules whose structures are NOT
in the 150k index (forced-zero exact-hit by construction). Measures the
precursor-window and fallback behavior of the Rung 1 predictor:
  - empty_windows: molecules with no index vector in +/-0.5 Da of any spectrum
  - fallback_only: molecules where every spectrum fell back to closest-match
Writes RESULT to models/coverage_result.txt as well as stdout.
"""
import os
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict

ROOT = os.path.expanduser('~/workspace/kaggle-casmi26')
N_BINS = 1500
PRECURSOR_TOL = 0.5
SIM_THRESHOLD = 0.2
TOP_K = 20


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


print("Loading index...", flush=True)
idx = np.load(os.path.join(ROOT, 'models/casmi_index.npz'))
precursors = idx['precursors']
vectors = idx['vectors'].astype(np.float32)
idx_smiles = set(str(s) for s in idx['smiles'])
print(f"Index: {len(precursors)} vectors", flush=True)

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
        continue  # index was built from the first ~200k rows
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


def predict(spectra):
    cand_scores = defaultdict(float)
    n_win = 0
    for prec, qvec in spectra:
        lo = np.searchsorted(precursors, prec - PRECURSOR_TOL, side='left')
        hi = np.searchsorted(precursors, prec + PRECURSOR_TOL, side='right')
        if hi <= lo:
            continue
        n_win += 1
        sims = vectors[lo:hi] @ qvec
        for j in np.argsort(sims)[::-1][:TOP_K]:
            s = float(sims[j])
            if s < SIM_THRESHOLD:
                break
            cand_scores[str(idx['smiles'][lo + j])] += s
    ranked = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
    top25 = [s for s, _ in ranked[:25]]
    if not top25:
        best, bs = None, -1
        for prec, qvec in spectra:
            lo = np.searchsorted(precursors, prec - PRECURSOR_TOL, side='left')
            hi = np.searchsorted(precursors, prec + PRECURSOR_TOL, side='right')
            if hi <= lo:
                continue
            sims = vectors[lo:hi] @ qvec
            bi = int(np.argmax(sims))
            if float(sims[bi]) > bs:
                bs = float(sims[bi])
                best = str(idx['smiles'][lo + bi])
        top25 = [best] * 25 if best else []
    while len(top25) < 25 and top25:
        top25.append(top25[0])
    return top25, n_win


hits1 = hits25 = n = 0
empty_win = 0
fb = 0
for i, (smi, spectra) in enumerate(held.items()):
    top25, nw = predict(spectra)
    if nw == 0:
        empty_win += 1
    if not top25:
        continue
    n += 1
    if top25[0] == smi:
        hits1 += 1
    if smi in top25:
        hits25 += 1
    if len(set(top25)) == 1:
        fb += 1
    if (i + 1) % 50 == 0:
        print(f"  scored {i+1}/{len(held)}", flush=True)

result = (f"RESULT scored={n} top1_rate={hits1/max(n,1):.3f} "
          f"top25_rate={hits25/max(n,1):.3f} empty_windows={empty_win} "
          f"fallback_only={fb}")
print(result, flush=True)
with open(os.path.join(ROOT, 'models/coverage_result.txt'), 'w') as f:
    f.write(result + "\n")
print("Wrote models/coverage_result.txt", flush=True)
