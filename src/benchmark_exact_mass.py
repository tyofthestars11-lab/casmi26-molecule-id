"""
Valid spectrum-holdout benchmark for the exact-mass scorer.

Design (fixes the forced-zero flaw of the earlier diagnostic):
  - reference set: raw peak lists, >=1 spectrum per molecule kept
  - query set: held-out spectra whose molecule IS in the reference set
Measures top-1 / top-5 / top-25 molecule retrieval.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict
from exact_mass_search import similarity

PRECURSOR_TOL = 0.5
MAX_REF = 20000
MAX_QUERY_MOLS = 300

ref_prec, ref_peaks, ref_mols = [], [], []
mol_to_spectra = defaultdict(list)

print("Reading spectra...", flush=True)
pf = pq.ParquetFile(os.path.expanduser('~/workspace/kaggle-casmi26/data/train.parquet'))
n_rows = 0
for batch in pf.iter_batches(batch_size=100000,
        columns=['ms2_mzs','ms2_normalized_intensities','precursor_mz',
                 'normalized_smiles','ingest_lib']):
    df = batch.to_pandas()
    df = df[df['ingest_lib'].isin(['enveda-np-examples','enveda-180'])]
    for _, row in df.iterrows():
        smi = str(row['normalized_smiles'])
        mol_to_spectra[smi].append((
            float(row['precursor_mz']),
            np.asarray(row['ms2_mzs'], dtype=np.float64),
            np.asarray(row['ms2_normalized_intensities'], dtype=np.float64),
        ))
    n_rows += len(df)
    if n_rows >= 300000:
        break
print(f"Rows read: {n_rows}, molecules: {len(mol_to_spectra)}", flush=True)

queries = []  # (true_smi, prec, mzs, ints)
for smi, specs in mol_to_spectra.items():
    if len(specs) >= 2 and len(queries) < MAX_QUERY_MOLS:
        queries.append((smi,) + specs[0])
        rest = specs[1:]
    else:
        rest = specs
    for s in rest:
        if len(ref_prec) >= MAX_REF:
            break
        ref_prec.append(s[0]); ref_peaks.append((s[1], s[2])); ref_mols.append(smi)
    if len(ref_prec) >= MAX_REF and len(queries) >= MAX_QUERY_MOLS:
        break

ref_prec = np.array(ref_prec)
order = np.argsort(ref_prec)
ref_prec = ref_prec[order]
ref_peaks = [ref_peaks[i] for i in order]
ref_mols = [ref_mols[i] for i in order]
print(f"Reference: {len(ref_prec)} spectra, Queries: {len(queries)} molecules", flush=True)

hits1 = hits5 = hits25 = 0
for qi, (true_smi, q_prec, q_mz, q_int) in enumerate(queries):
    lo = np.searchsorted(ref_prec, q_prec - PRECURSOR_TOL, side='left')
    hi = np.searchsorted(ref_prec, q_prec + PRECURSOR_TOL, side='right')
    best = {}
    for j in range(lo, hi):
        r_mz, r_int = ref_peaks[j]
        s = similarity(q_prec, q_mz, q_int, ref_prec[j], r_mz, r_int)
        m = ref_mols[j]
        if s > best.get(m, 0):
            best[m] = s
    ranked = sorted(best, key=best.get, reverse=True)
    if true_smi in ranked[:1]: hits1 += 1
    if true_smi in ranked[:5]: hits5 += 1
    if true_smi in ranked[:25]: hits25 += 1
    if (qi + 1) % 50 == 0:
        print(f"  {qi+1}/{len(queries)} queries scored", flush=True)

n = len(queries)
print(f"RESULT n={n} top1={hits1/n:.3f} top5={hits5/n:.3f} top25={hits25/n:.3f}", flush=True)
