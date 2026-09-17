import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict
import time
import sys

N_BINS = 1500
BIN_WIDTH = 1.0

def spectrum_to_vector(mzs, intensities, precursor_mz):
    """Bin spectrum into fixed-size vector."""
    vec = np.zeros(N_BINS, dtype=np.float32)
    mzs = np.array(mzs)
    intensities = np.array(intensities)
    mask = (intensities >= 0.01) & (mzs <= precursor_mz + 2.0) & (mzs < N_BINS)
    mzs, intensities = mzs[mask], intensities[mask]
    bins = (mzs / BIN_WIDTH).astype(int)
    bins = np.clip(bins, 0, N_BINS-1)
    for b, inten in zip(bins, intensities):
        vec[b] += inten
    vec = np.sqrt(vec)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()

log("Loading test...")
test_df = pd.read_parquet('data/test.parquet')
tmin, tmax = test_df['precursor_mz'].min() - 1.0, test_df['precursor_mz'].max() + 1.0
log(f"Test: {len(test_df)} spectra, precursor {tmin:.1f}-{tmax:.1f}")

log("Vectorizing test...")
test_vecs = []
test_info = []
for _, row in test_df.iterrows():
    try:
        vec = spectrum_to_vector(row['ms2_mzs'], row['ms2_normalized_intensities'], row['precursor_mz'])
        test_vecs.append(vec)
        test_info.append((row['molecule_id'], row['precursor_mz'], row['base_peak_intensity'] or 1.0))
    except:
        continue
test_vecs = np.array(test_vecs)
log(f"Test vectors: {test_vecs.shape}")

log("Streaming train...")
train_vecs = []
train_smiles = []
train_prec = []
t0 = time.time()
pf = pq.ParquetFile('data/train.parquet')
for batch in pf.iter_batches(batch_size=100000, columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'normalized_smiles', 'ingest_lib']):
    df = batch.to_pandas()
    df = df[(df['precursor_mz'] >= tmin) & (df['precursor_mz'] <= tmax)]
    df = df[df['ingest_lib'].isin(['enveda-np-examples', 'enveda-180'])]
    for _, row in df.iterrows():
        try:
            vec = spectrum_to_vector(row['ms2_mzs'], row['ms2_normalized_intensities'], row['precursor_mz'])
            if np.sum(vec) > 0:
                train_vecs.append(vec)
                train_smiles.append(row['normalized_smiles'])
                train_prec.append(row['precursor_mz'])
        except:
            continue
    if len(train_vecs) >= 80000:
        break
    log(f"  {len(train_vecs)} vectors, {time.time()-t0:.0f}s")

train_vecs = np.array(train_vecs)
train_prec = np.array(train_prec)
log(f"Train vectors: {train_vecs.shape}")

log("Computing similarities...")
mol_candidates = defaultdict(lambda: defaultdict(float))
chunk_size = 200
for start in range(0, len(test_vecs), chunk_size):
    end = min(start + chunk_size, len(test_vecs))
    sims = test_vecs[start:end] @ train_vecs.T
    for i in range(end - start):
        mol_id, tprec, bp = test_info[start + i]
        prec_mask = np.abs(train_prec - tprec) <= 0.5
        sim_row = sims[i]
        sim_row[~prec_mask] = 0
        top_idx = np.argsort(sim_row)[-20:][::-1]
        for idx in top_idx:
            if sim_row[idx] > 0.2:
                mol_candidates[mol_id][train_smiles[idx]] += float(sim_row[idx]) * bp
    log(f"  {end}/{len(test_vecs)}, {time.time()-t0:.0f}s")

log("Writing submission...")
with open('submissions/baseline_v1.csv', 'w') as f:
    f.write('molecule_id,' + ','.join(f'smiles_{i+1}' for i in range(25)) + '\n')
    for mol_id in test_df['molecule_id'].unique():
        cands = mol_candidates.get(mol_id, {})
        ranked = sorted(cands.items(), key=lambda x: -x[1])[:25]
        sl = [s for s, _ in ranked]
        while len(sl) < 25:
            sl.append(sl[0] if sl else "")
        f.write(mol_id + ',' + ','.join(f'"{s}"' for s in sl) + '\n')
log(f"DONE {time.time()-t0:.0f}s")
