"""
Rung 2 benchmark: phi-frame vs exact-mass baseline, same valid holdout.

Reports raw:
  - per-map top-1 accuracy (map1 raw, map2 rung, map3 conjugate, map4 reflection)
  - phi-frame (map2, nearest neighbor by rung address) top-1/5/25
  - exact-mass baseline top-1/5/25 on the identical queries
  - 4-way lock: LOCKED/MARGINAL/REJECTED rates + accuracy when locked
  - divergence: where phi-frame and baseline disagree, who was right
  - novel candidates: queries whose best rung-similarity falls below threshold
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict

from spectral_tiled_lucas import rung_histogram, histogram_cosine, rung_address
from spectral_fc_chars import (rung_signature, signature_cosine,
                               interval_sequence, interval_match,
                               reflection_symmetric, symmetric_cosine)
from spectral_lock4way import (map1_vector, map1_score, lock_verdict, MAPS)
from exact_mass_search import similarity as exact_mass_sim

ROOT = os.path.expanduser('~/workspace/kaggle-casmi26')
PRECURSOR_TOL = 0.5
MAX_REF = 20000
MAX_QUERY_MOLS = 300
NOVEL_THR = 0.3


def prep(spec):
    prec, mzs, ints = spec
    q = (mzs, ints, prec)
    return {
        'mol': None,
        'map1': map1_vector(mzs, ints, prec),
        'map2': rung_histogram(mzs, ints, prec),
        'map3': interval_sequence(mzs, ints, prec),
        'map4': (reflection_symmetric(
            rung_signature(mzs, ints, prec),
            float(rung_address(np.array([prec]))[0])), prec),
        'exact': q,
    }


print("Reading spectra...", flush=True)
pf = pq.ParquetFile(os.path.join(ROOT, 'data/train.parquet'))
mol_to_spectra = defaultdict(list)
n_rows = 0
for batch in pf.iter_batches(batch_size=100000,
        columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz',
                 'normalized_smiles', 'ingest_lib']):
    df = batch.to_pandas()
    df = df[df['ingest_lib'].isin(['enveda-np-examples', 'enveda-180'])]
    for _, row in df.iterrows():
        mol_to_spectra[str(row['normalized_smiles'])].append((
            float(row['precursor_mz']),
            np.asarray(row['ms2_mzs'], dtype=np.float64),
            np.asarray(row['ms2_normalized_intensities'], dtype=np.float64)))
    n_rows += len(df)
    if n_rows >= 300000:
        break

ref, queries = [], []
for smi, specs in mol_to_spectra.items():
    if len(specs) >= 2 and len(queries) < MAX_QUERY_MOLS:
        queries.append((smi, specs[0]))
        rest = specs[1:]
    else:
        rest = specs
    for s in rest:
        if len(ref) >= MAX_REF:
            break
        ref.append((smi, s))
    if len(ref) >= MAX_REF and len(queries) >= MAX_QUERY_MOLS:
        break

print("Prepping reference representations...", flush=True)
ref_prep = []
for smi, s in ref:
    p = prep(s)
    p['mol'] = smi
    p['prec'] = s[0]
    ref_prep.append(p)
ref_prec = np.array([p['prec'] for p in ref_prep])
order = np.argsort(ref_prec)
ref_prep = [ref_prep[i] for i in order]
ref_prec = ref_prec[order]
print(f"Reference: {len(ref_prep)} spectra, Queries: {len(queries)}", flush=True)


def map_scores(qp, cands):
    """Score one query against window candidates under all 4 maps + baseline."""
    out = {k: {} for k in ['map1', 'map2', 'map3', 'map4', 'exact']}
    for rp in cands:
        m = rp['mol']
        s1 = map1_score(qp['map1'], rp['map1'])
        if s1 > out['map1'].get(m, 0): out['map1'][m] = s1
        qc, qw = qp['map2']; rc, rw = rp['map2']
        s2 = histogram_cosine(qc, qw, rc, rw)
        if s2 > out['map2'].get(m, 0): out['map2'][m] = s2
        q_iv, q_w = qp['map3']; r_iv, r_w = rp['map3']
        fwd = interval_match(q_iv, q_w, r_iv, r_w)
        conj = interval_match(q_iv[::-1], q_w[::-1], r_iv[::-1], r_w[::-1]) \
            if len(q_iv) and len(r_iv) else 0.0
        s3 = 0.5 * (fwd + conj)
        if s3 > out['map3'].get(m, 0): out['map3'][m] = s3
        s4 = symmetric_cosine(qp['map4'][0], rp['map4'][0])
        if s4 > out['map4'].get(m, 0): out['map4'][m] = s4
        se = exact_mass_sim(qp['exact'][2], qp['exact'][0], qp['exact'][1],
                            rp['prec'], rp['exact'][0], rp['exact'][1])
        if se > out['exact'].get(m, 0): out['exact'][m] = se
    return out


def top1(d):
    return max(d, key=d.get) if d else None


map_hits = {k: 0 for k in ['map1', 'map2', 'map3', 'map4']}
phi_hits = [0, 0, 0]  # top1, top5, top25 for map2 ranking
base_hits = [0, 0, 0]
lock_tally = {'LOCKED': 0, 'MARGINAL': 0, 'REJECTED': 0}
lock_correct = {'LOCKED': 0, 'MARGINAL': 0, 'REJECTED': 0}
diverge = {'agree': 0, 'phi_right': 0, 'base_right': 0, 'neither': 0}
novel = 0
novel_sims = []

for qi, (true_smi, qs) in enumerate(queries):
    qp = prep(qs)
    q_prec = qs[0]
    lo = np.searchsorted(ref_prec, q_prec - PRECURSOR_TOL, side='left')
    hi = np.searchsorted(ref_prec, q_prec + PRECURSOR_TOL, side='right')
    cands = ref_prep[lo:hi]
    sc = map_scores(qp, cands)

    votes = [top1(sc[k]) for k in ['map1', 'map2', 'map3', 'map4']]
    verdict = lock_verdict(votes)
    lock_tally[verdict] += 1
    if votes[0] == true_smi:
        lock_correct[verdict] += 1

    for k in ['map1', 'map2', 'map3', 'map4']:
        if top1(sc[k]) == true_smi:
            map_hits[k] += 1

    phi_ranked = sorted(sc['map2'], key=sc['map2'].get, reverse=True)
    base_ranked = sorted(sc['exact'], key=sc['exact'].get, reverse=True)
    for i, ranked in ((0, phi_ranked), (1, base_ranked)):
        pass
    if true_smi in phi_ranked[:1]: phi_hits[0] += 1
    if true_smi in phi_ranked[:5]: phi_hits[1] += 1
    if true_smi in phi_ranked[:25]: phi_hits[2] += 1
    if true_smi in base_ranked[:1]: base_hits[0] += 1
    if true_smi in base_ranked[:5]: base_hits[1] += 1
    if true_smi in base_ranked[:25]: base_hits[2] += 1

    pv, bv = (phi_ranked[0] if phi_ranked else None,
              base_ranked[0] if base_ranked else None)
    if pv == bv:
        diverge['agree'] += 1
    elif pv == true_smi:
        diverge['phi_right'] += 1
    elif bv == true_smi:
        diverge['base_right'] += 1
    else:
        diverge['neither'] += 1

    best_phi = sc['map2'].get(pv, 0.0) if pv else 0.0
    novel_sims.append(best_phi)
    if best_phi < NOVEL_THR:
        novel += 1

    if (qi + 1) % 50 == 0:
        print(f"  {qi+1}/{len(queries)}", flush=True)

n = len(queries)
print("=" * 60, flush=True)
print(f"n={n}", flush=True)
for k in ['map1', 'map2', 'map3', 'map4']:
    print(f"  {k} top1={map_hits[k]/n:.3f}", flush=True)
print(f"  phi-frame(map2) top1={phi_hits[0]/n:.3f} "
      f"top5={phi_hits[1]/n:.3f} top25={phi_hits[2]/n:.3f}", flush=True)
print(f"  exact-mass      top1={base_hits[0]/n:.3f} "
      f"top5={base_hits[1]/n:.3f} top25={base_hits[2]/n:.3f}", flush=True)
for v in ['LOCKED', 'MARGINAL', 'REJECTED']:
    t = lock_tally[v]
    print(f"  lock {v}: {t} ({t/n:.3f}), accuracy when {v}: "
          f"{lock_correct[v]/max(t,1):.3f}", flush=True)
print(f"  divergence: agree={diverge['agree']} phi_right={diverge['phi_right']} "
      f"base_right={diverge['base_right']} neither={diverge['neither']}", flush=True)
print(f"  novel candidates (best rung-sim < {NOVEL_THR}): {novel}/{n}", flush=True)
print(f"  novel sim distribution: min={min(novel_sims):.3f} "
      f"median={np.median(novel_sims):.3f}", flush=True)
