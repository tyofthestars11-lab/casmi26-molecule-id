"""
Rung 2 corrected benchmark — never Cartesian to measure non-Cartesian.

v1 measured the phi-rung map with histogram_cosine: a fixed Cartesian grid
(integer bins) + a dot product. That is Cartesian machinery pointed at
phi coordinates. v2 removes the grid entirely.

Map 2 (phi-rung, phi-native): spectra are sets of rung addresses
r = ln(mz)/ln(phi) with intensity weights. Matching is address-based:
greedy intensity-ordered nearest-rung matching with a CONSTANT rung
tolerance. Key structural fact: fixed ppm in m/z  <=>  fixed delta-rung,
because dr = (dm/m)/ln(phi). The rung coordinate absorbs the scale, so
one tolerance number works at every mass. No bins. No dot product.

Map 3 (conjugate intervals, phi-native): adjacent-peak spacings in rung
units are scale-free by construction (ratio of masses -> difference of
rungs). Forward read = positional interval alignment query-asc vs ref-asc.
Reverse/conjugate read = query-asc vs ref-desc. Genuinely positional,
genuinely independent of map 2 (spacings, not positions).

Map 1 stays Cartesian on purpose: the 1-Da binned cosine is the Cartesian
reference point, labeled as such. Map 4 (reflection) is dropped — domain
fact from v1: fragmentation has no reflection symmetry about the precursor.

Lock: LOCKED = map1+map2+map3 agree; MARGINAL = 2 of 3; REJECTED = <2.
Novel flag: 5th percentile of the map-2 top-score distribution (relative,
not absolute).
"""

import os
import sys
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from spectral_lock4way import map1_vector, map1_score          # Cartesian baseline (labeled)
from exact_mass_search import similarity as exact_mass_sim     # chemistry baseline

PHI = (1 + 5 ** 0.5) / 2
LN_PHI = float(np.log(PHI))
PPM = 20.0
TOL_RUNG = PPM * 1e-6 / LN_PHI      # 4.155e-5 rungs: the data's own 20ppm resolution, in rungs
TOL_INT = 2.0 * TOL_RUNG
INT_FLOOR = 0.01
PRECURSOR_TOL = 0.5
MAX_REF = 20000
MAX_QUERY_MOLS = 300


def rung_address(mz):
    return np.log(np.asarray(mz, dtype=np.float64)) / LN_PHI


def prep(spec):
    """Phi-native representation: rung addresses + interval structure. No grid."""
    prec, mzs, intens = spec
    m = np.asarray(mzs, dtype=np.float64)
    it = np.asarray(intens, dtype=np.float64)
    mask = (it >= INT_FLOOR) & (m <= prec + 2.0) & (m > 0)
    m, it = m[mask], it[mask]
    if len(m) == 0:
        return None
    o = np.argsort(m)
    m, it = m[o], it[o]
    r = rung_address(m)
    od = np.argsort(-it)                       # intensity-desc order (greedy)
    if len(r) > 1:
        iv = np.diff(r)                        # rung intervals: scale-free spacings
        iw = np.sqrt(it[:-1] * it[1:])
        od_iv = np.argsort(-iw)
    else:
        iv, iw, od_iv = np.array([]), np.array([]), np.array([], dtype=int)
    return {
        'prec': float(prec),
        'r': r, 'it': it, 'od': od,
        'nq': float(np.sqrt(it.sum())),
        'iv': iv, 'iw': iw, 'od_iv': od_iv,
        'niw': float(np.sqrt(iw.sum())) if len(iw) else 0.0,
        'map1': map1_vector(mzs, intens, prec),   # Cartesian reference only
        'exact': (np.asarray(mzs, dtype=np.float64),
                  np.asarray(intens, dtype=np.float64), float(prec)),
    }


def map2_rung_match(q, rp):
    """Address-based nearest-rung matching. No bins, no dot product."""
    if q is None or rp is None:
        return 0.0
    rr, rit = rp['r'], rp['it']
    used = np.zeros(len(rr), dtype=bool)
    s = 0.0
    for qi in q['od']:
        rq = q['r'][qi]
        j = int(np.searchsorted(rr, rq))
        best, bd = -1, TOL_RUNG
        jj = j - 1
        if 0 <= jj < len(rr) and not used[jj]:
            d = abs(rr[jj] - rq)
            if d <= bd:
                bd, best = d, jj
        jj = j
        if 0 <= jj < len(rr) and not used[jj]:
            d = abs(rr[jj] - rq)
            if d <= bd:
                bd, best = d, jj
        if best >= 0:
            used[best] = True
            s += float(np.sqrt(q['it'][qi] * rit[best]))
    if q['nq'] == 0 or rp['nq'] == 0:
        return 0.0
    return s / (q['nq'] * rp['nq'])


def _align(qiv, qiw, riv, riw):
    """Greedy two-pointer positional interval alignment (both ascending)."""
    i = j = 0
    s = 0.0
    n, m = len(qiv), len(riv)
    while i < n and j < m:
        if abs(qiv[i] - riv[j]) <= TOL_INT:
            s += float(np.sqrt(qiw[i] * riw[j]))
            i += 1
            j += 1
        elif qiv[i] < riv[j]:
            i += 1
        else:
            j += 1
    return s


MAP3_TOL = 1e-4  # v1 witness tolerance in rung units (~48 ppm on intervals)


def map3_multiset(q, rp):
    """Conjugate-interval witness: greedy weight-ordered multiset matching of
    rung intervals. v2's positional alignment measured 0.423 (too brittle:
    one dropped peak shifts every downstream position); the multiset is the
    robust witness (v1: 0.723). v1's double-reverse is dropped as vacuous
    (reversing both sequences of an order-insensitive matcher cancels)."""
    if q is None or rp is None or len(q['iv']) == 0 or len(rp['iv']) == 0:
        return 0.0
    if q['niw'] == 0 or rp['niw'] == 0:
        return 0.0
    used = np.zeros(len(rp['iv']), dtype=bool)
    num = 0.0
    for i in q['od_iv']:
        d = np.abs(rp['iv'] - q['iv'][i])
        cand = np.where((d <= MAP3_TOL) & (~used))[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmin(d[cand])]
        used[j] = True
        num += float(np.sqrt(q['iw'][i] * rp['iw'][j]))
    den = q['niw'] * rp['niw']
    return num / den if den > 0 else 0.0


# ---- self-test: identical spectra must score ~1.0 under both phi maps ----
def _self_test():
    m = np.array([100.1234, 150.5678, 200.1111, 300.9999])
    it = np.array([1.0, 0.5, 0.8, 0.3])
    a = prep((450.0, m, it))
    b = prep((450.0, m, it))
    s2 = map2_rung_match(a, b)
    s3 = map3_multiset(a, b)
    print(f"self-test identical spectra: map2={s2:.4f} map3={s3:.4f}", flush=True)
    assert s2 > 0.99, f"map2 self-test failed: {s2}"
    assert s3 > 0.99, f"map3 self-test failed: {s3}"
    # shifted copy (all peaks +50 ppm) must still match under rung tol? 50ppm > 20ppm tol -> weak
    c = prep((450.0, m * 1.00005, it))
    s2s = map2_rung_match(a, c)
    print(f"self-test +50ppm shift: map2={s2s:.4f} (expect < 0.5)", flush=True)
    assert s2s < 0.5, f"map2 shift test failed: {s2s}"
    print("self-test PASSED", flush=True)


print("=== Rung 2 corrected benchmark (v2): no Cartesian measure of phi maps (v3 run) ===", flush=True)
print(f"TOL_RUNG = {TOL_RUNG:.3e} rungs  (= {PPM} ppm, scale-free)", flush=True)
_self_test()

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

print("Prepping phi-native representations...", flush=True)
ref_prep = []
for smi, s in ref:
    p = prep(s)
    if p is None:
        continue
    p['mol'] = smi
    ref_prep.append(p)
ref_prec = np.array([p['prec'] for p in ref_prep])
order = np.argsort(ref_prec)
ref_prep = [ref_prep[i] for i in order]
ref_prec = ref_prec[order]
print(f"Reference: {len(ref_prep)} spectra, Queries: {len(queries)}", flush=True)


def score_all(qp, cands):
    out = {k: {} for k in ['map1', 'map2', 'map3', 'exact']}
    for rp in cands:
        m = rp['mol']
        s1 = map1_score(qp['map1'], rp['map1'])
        if s1 > out['map1'].get(m, 0):
            out['map1'][m] = s1
        s2 = map2_rung_match(qp, rp)
        if s2 > out['map2'].get(m, 0):
            out['map2'][m] = s2
        s3 = map3_multiset(qp, rp)
        if s3 > out['map3'].get(m, 0):
            out['map3'][m] = s3
        qe = qp['exact']
        se = exact_mass_sim(qe[2], qe[0], qe[1], rp['prec'],
                            rp['exact'][0], rp['exact'][1])
        if se > out['exact'].get(m, 0):
            out['exact'][m] = se
    return out


def top1(d):
    return max(d, key=d.get) if d else None


def topk(d, k):
    return sorted(d, key=d.get, reverse=True)[:k]


map_hits = {k: 0 for k in ['map1', 'map2', 'map3']}
phi_hits = [0, 0, 0]
base_hits = [0, 0, 0]
lock_tally = {'LOCKED': 0, 'MARGINAL': 0, 'REJECTED': 0}
lock_correct = {'LOCKED': 0, 'MARGINAL': 0, 'REJECTED': 0}
lock_outputs = 0
rejected_fallback_hits = 0
rejected_fallback_n = 0
diverge = {'agree': 0, 'phi_right': 0, 'base_right': 0, 'neither': 0}
novel_sims = []

for qi, (true_smi, qs) in enumerate(queries):
    qp = prep(qs)
    if qp is None:
        continue
    q_prec = qs[0]
    lo = np.searchsorted(ref_prec, q_prec - PRECURSOR_TOL, side='left')
    hi = np.searchsorted(ref_prec, q_prec + PRECURSOR_TOL, side='right')
    cands = ref_prep[lo:hi]
    sc = score_all(qp, cands)
    t = {k: top1(sc[k]) for k in sc}
    for k in map_hits:
        if t[k] == true_smi:
            map_hits[k] += 1
    for i, k in enumerate([1, 5, 25]):
        if true_smi in topk(sc['map2'], k):
            phi_hits[i] += 1
        if true_smi in topk(sc['exact'], k):
            base_hits[i] += 1
    votes = [t['map1'], t['map2'], t['map3']]
    uniq = set(votes)
    if len(uniq) == 1:
        tier = 'LOCKED'
        out = votes[0]
    elif len(uniq) == 2:
        tier = 'MARGINAL'
        out = max(uniq, key=votes.count)
    else:
        tier = 'REJECTED'
        out = None
    lock_tally[tier] += 1
    if out is not None:
        lock_outputs += 1
        if out == true_smi:
            lock_correct[tier] += 1
    else:
        rejected_fallback_n += 1
        if t['map1'] == true_smi:
            rejected_fallback_hits += 1
    # divergence: phi-rung (map2) vs chemistry baseline
    pe, be = (t['map2'] == true_smi), (t['exact'] == true_smi)
    if t['map2'] == t['exact']:
        diverge['agree'] += 1
    elif pe and not be:
        diverge['phi_right'] += 1
    elif be and not pe:
        diverge['base_right'] += 1
    else:
        diverge['neither'] += 1
    if sc['map2']:
        novel_sims.append(max(sc['map2'].values()))
    if (qi + 1) % 50 == 0:
        print(f"  ... {qi + 1}/{len(queries)}", flush=True)

nq = len(queries)
novel_sims = np.array(novel_sims)
nov_thr = float(np.percentile(novel_sims, 5))
n_flagged = int((novel_sims < nov_thr).sum())

L = []
L.append("RUNG 2 CORRECTED BENCHMARK (v3) — 2026-09-17")
L.append("map3 = conjugate-interval MULTISET (v1 witness 0.723; vacuous double-reverse removed).")
L.append("v2 positional alignment measured 0.423 and is recorded as a negative: too brittle.")
L.append("No Cartesian measure of non-Cartesian maps. map4 dropped (domain fact).")
L.append(f"TOL_RUNG = {TOL_RUNG:.3e} rungs = {PPM} ppm, scale-free (one number, every mass)")
L.append(f"Reference spectra: {len(ref_prep)}, Queries: {nq}")
L.append("")
L.append("MAP TOP-1 (known-structure holdout):")
for k in ['map1', 'map2', 'map3']:
    label = {'map1': 'map1 raw binned (Cartesian reference)',
             'map2': 'map2 phi-rung address match (phi-native)',
             'map3': 'map3 conjugate intervals (phi-native)'}[k]
    L.append(f"  {label}: {map_hits[k] / nq:.3f}")
L.append("")
L.append(f"PHI-RUNG (map2) top1/top5/top25: "
         f"{phi_hits[0] / nq:.3f} / {phi_hits[1] / nq:.3f} / {phi_hits[2] / nq:.3f}")
L.append(f"EXACT-MASS top1/top5/top25:      "
         f"{base_hits[0] / nq:.3f} / {base_hits[1] / nq:.3f} / {base_hits[2] / nq:.3f}")
L.append("")
L.append("LOCK (3 witnesses: map1+map2+map3):")
for tier in ['LOCKED', 'MARGINAL', 'REJECTED']:
    n = lock_tally[tier]
    acc = lock_correct[tier] / n if n else 0.0
    L.append(f"  {tier}: {n}/{nq} ({n / nq * 100:.1f}%)  output-accuracy {lock_correct[tier]}/{n} = {acc:.3f}")
L.append(f"  REJECTED fallback (map1 read, labeled): "
         f"{rejected_fallback_hits}/{rejected_fallback_n} = "
         f"{rejected_fallback_hits / rejected_fallback_n:.3f}" if rejected_fallback_n else "")
L.append("")
L.append("DIVERGENCE map2 (phi-rung) vs exact-mass (chemistry):")
L.append(f"  agree {diverge['agree']}, phi-right {diverge['phi_right']}, "
         f"base-right {diverge['base_right']}, neither {diverge['neither']}")
L.append("")
L.append("NOVEL FLAG (relative threshold):")
L.append(f"  map2 top-score: min {novel_sims.min():.3f}, 5th-pct {nov_thr:.3f}, "
         f"median {np.median(novel_sims):.3f}")
L.append(f"  flagged below 5th percentile: {n_flagged}/{nq}")
L.append("  NOTE: holdout queries all have their molecule in-ref; flags are")
L.append("  low-confidence calls, not true novels. True novel test needs the")
L.append("  molecule-disjoint design.")
report = "\n".join(L)
print(report, flush=True)
with open(os.path.join(ROOT, 'models', 'phi_frame_v3_report.txt'), 'w') as f:
    f.write(report + "\n")
print("report -> models/phi_frame_v3_report.txt", flush=True)
