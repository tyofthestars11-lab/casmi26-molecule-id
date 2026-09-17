"""
Build the deployable raw-peak reference index for CASMI 2026 submissions.

Molecule-balanced (not first-N): up to 3 spectra per molecule across the
full Enveda corpus, streamed in chunks — the 3GB train.parquet is never
fully loaded. Raw peaks (not bins) so the notebook can run exact-mass
ppm matching + the phi-frame maps at the data's own 4-decimal resolution.

Output: models/casmi_raw_index.npz
  precursors  (N,) float32, sorted ascending
  smiles      (N,) unicode
  adduct      (N,) unicode
  p_starts    (N,) int64     offsets into flat peak arrays
  p_lens      (N,) int64
  flat_mz     (M,) float32
  flat_int    (M,) float16
  sha256 in models/casmi_raw_index.sha256
"""

import os
import hashlib
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, 'data', 'train.parquet')
OUT = os.path.join(ROOT, 'models', 'casmi_raw_index.npz')

PER_MOL = 3
INT_FLOOR = 0.01
MAX_PEAKS = 300
LIBS = ('enveda-np-examples', 'enveda-180')


def clean(prec, mzs, intens):
    m = np.asarray(mzs, dtype=np.float64)
    it = np.asarray(intens, dtype=np.float64)
    mask = (it >= INT_FLOOR) & (m <= prec + 2.0) & (m > 0)
    m, it = m[mask], it[mask]
    if len(m) == 0:
        return None, None
    if len(m) > MAX_PEAKS:
        keep = np.argsort(-it)[:MAX_PEAKS]
        m, it = m[keep], it[keep]
    o = np.argsort(m)
    return m[o].astype(np.float32), it[o].astype(np.float16)


print("Streaming train.parquet (molecule-balanced, 3/mol)...", flush=True)
kept = defaultdict(list)   # smiles -> list of (prec, mz, inten, adduct)
n_rows = 0
pf = pq.ParquetFile(TRAIN)
for batch in pf.iter_batches(batch_size=100000,
        columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz',
                 'normalized_smiles', 'ingest_lib', 'adduct']):
    df = batch.to_pandas()
    df = df[df['ingest_lib'].isin(LIBS)]
    for _, row in df.iterrows():
        smi = str(row['normalized_smiles'])
        if len(kept[smi]) >= PER_MOL:
            continue
        prec = float(row['precursor_mz'])
        mz, it = clean(prec, row['ms2_mzs'], row['ms2_normalized_intensities'])
        if mz is None:
            continue
        kept[smi].append((prec, mz, it, str(row.get('adduct', ''))))
    n_rows += len(df)
    if n_rows % 500000 < 100000:
        print(f"  rows scanned: {n_rows}, molecules: {len(kept)}", flush=True)

print(f"Scanned {n_rows} rows -> {len(kept)} molecules", flush=True)

precs, smiles, adducts, mzs, intens = [], [], [], [], []
for smi, specs in kept.items():
    for prec, mz, it, ad in specs:
        precs.append(prec)
        smiles.append(smi)
        adducts.append(ad)
        mzs.append(mz)
        intens.append(it)

precs = np.array(precs, dtype=np.float32)
order = np.argsort(precs)
precs = precs[order]
smiles = np.array([smiles[i] for i in order])
adducts = np.array([adducts[i] for i in order])
mzs = [mzs[i] for i in order]
intens = [intens[i] for i in order]

starts = np.zeros(len(precs), dtype=np.int64)
lens = np.zeros(len(precs), dtype=np.int64)
pos = 0
for i, mz in enumerate(mzs):
    starts[i] = pos
    lens[i] = len(mz)
    pos += len(mz)
flat_mz = np.concatenate(mzs).astype(np.float32)
flat_int = np.concatenate(intens).astype(np.float16)

print(f"Spectra: {len(precs)}, total peaks: {len(flat_mz)}", flush=True)
np.savez_compressed(OUT,
                    precursors=precs, smiles=smiles, adduct=adducts,
                    p_starts=starts, p_lens=lens,
                    flat_mz=flat_mz, flat_int=flat_int)
size = os.path.getsize(OUT)
h = hashlib.sha256()
with open(OUT, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        h.update(chunk)
with open(os.path.join(ROOT, 'models', 'casmi_raw_index.sha256'), 'w') as f:
    f.write(f"{h.hexdigest()}  casmi_raw_index.npz\n")
print(f"Wrote {OUT} ({size / 1e6:.1f} MB)  sha256={h.hexdigest()[:16]}...", flush=True)
