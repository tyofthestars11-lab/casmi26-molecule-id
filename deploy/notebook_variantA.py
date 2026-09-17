"""
CASMI 2026 — Rung 2 deployment: exact-mass chemistry ranking + phi-frame lock.

Variant A (USE_LOCK=False): exact-mass baseline (0.7 frag + 0.3 neutral-loss,
20 ppm greedy matching) over a molecule-balanced raw-peak index. Top-25
SMILES per molecule by aggregated score.

Variant B (USE_LOCK=True): same ranking, plus the 3-witness lock
(map1 binned / map2 rung-address / map3 conjugate intervals). Where >=2
witnesses agree on a candidate above the baseline top-1, the lock promotes
it to rank 1. The lock names the confidence; the baseline names the read.

Internet OFF. Reads Kaggle runtime test.parquet (handles hidden IDs).
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import os

INPUT_DIR = os.environ.get('CASMI_INPUT',
    '/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra')
INDEX_PATH = os.environ.get('CASMI_INDEX', '/kaggle/input/casmi26-raw-index/casmi_raw_index.npz')
OUT_PATH = os.environ.get('CASMI_OUT', '/kaggle/working/submission.csv')

USE_LOCK = False

PRECURSOR_TOL = 0.5
PPM = 20.0
INT_FLOOR = 0.01
FRAG_W, LOSS_W = 0.7, 0.3
PHI = (1 + 5 ** 0.5) / 2
LN_PHI = float(np.log(PHI))
TOL_RUNG = PPM * 1e-6 / LN_PHI
MAP3_TOL = 1e-4
N_BINS = 1500
TOP_K = 25
MAX_QPEAKS = 400


# ---------- exact-mass chemistry ----------
def _match(q_mz, q_int, r_mz, r_int, ppm):
    # Greedy intensity-desc matching, nearest unused ref peak within ppm.
    # r_mz is sorted ascending (index invariant) -> searchsorted windows.
    # Exactly equivalent to the benchmarked full-scan version.
    if len(q_mz) == 0 or len(r_mz) == 0:
        return 0.0
    used = np.zeros(len(r_mz), dtype=bool)
    score = 0.0
    tols = q_mz * ppm * 1e-6
    los = np.searchsorted(r_mz, q_mz - tols)
    his = np.searchsorted(r_mz, q_mz + tols)
    for qi in range(len(q_mz)):
        best, bd = -1, tols[qi]
        lo, hi = int(los[qi]), int(his[qi])
        for j in range(lo, hi):
            if not used[j]:
                d = abs(r_mz[j] - q_mz[qi])
                if d < bd:
                    bd, best = d, j
        if best >= 0:
            used[best] = True
            score += float(np.sqrt(q_int[qi] * r_int[best]))
    return score


def _cos_match(s, q_int, r_int):
    nq, nr = float(np.sqrt(q_int.sum())), float(np.sqrt(r_int.sum()))
    return s / (nq * nr) if nq and nr else 0.0


def exact_mass_sim(q_prec, q_mz, q_int, r_prec, r_mz, r_int):
    frag = _cos_match(_match(q_mz, q_int, r_mz, r_int, PPM), q_int, r_int)
    q_lm = q_prec - q_mz
    qm = q_lm > 0
    r_lm = r_prec - r_mz
    rm = r_lm > 0
    loss = _cos_match(_match(q_lm[qm], q_int[qm], r_lm[rm], r_int[rm], PPM),
                      q_int[qm], r_int[rm])
    return FRAG_W * frag + LOSS_W * loss


# ---------- phi-frame witnesses ----------
def map1_binned(mz, it, prec):
    v = np.zeros(N_BINS, dtype=np.float32)
    m = (it >= INT_FLOOR) & (mz <= prec + 2.0) & (mz < N_BINS)
    b = np.clip(mz[m].astype(int), 0, N_BINS - 1)
    np.add.at(v, b, np.sqrt(it[m].astype(np.float32)))
    n = float(np.linalg.norm(v))
    return v / n if n else v


def rung_match(q_r, q_it, r_r, r_it):
    used = np.zeros(len(r_r), dtype=bool)
    s = 0.0
    for qi in np.argsort(-q_it):
        rq = q_r[qi]
        j = int(np.searchsorted(r_r, rq))
        best, bd = -1, TOL_RUNG
        for jj in (j - 1, j):
            if 0 <= jj < len(r_r) and not used[jj]:
                d = abs(r_r[jj] - rq)
                if d <= bd:
                    bd, best = d, jj
        if best >= 0:
            used[best] = True
            s += float(np.sqrt(q_it[qi] * r_it[best]))
    nq, nr = float(np.sqrt(q_it.sum())), float(np.sqrt(r_it.sum()))
    return s / (nq * nr) if nq and nr else 0.0


def interval_multiset(q_r, q_it, r_r, r_it):
    if len(q_r) < 2 or len(r_r) < 2:
        return 0.0
    q_iv, r_iv = np.diff(q_r), np.diff(r_r)
    q_iw = np.sqrt(q_it[:-1] * q_it[1:])
    r_iw = np.sqrt(r_it[:-1] * r_it[1:])
    used = np.zeros(len(r_iv), dtype=bool)
    num = 0.0
    for i in np.argsort(-q_iw):
        d = np.abs(r_iv - q_iv[i])
        cand = np.where((d <= MAP3_TOL) & (~used))[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmin(d[cand])]
        used[j] = True
        num += float(np.sqrt(q_iw[i] * r_iw[j]))
    den = float(np.sqrt(q_iw.sum()) * np.sqrt(r_iw.sum()))
    return num / den if den else 0.0


# ---------- query prep: merge a molecule's spectra ----------
def merge_query(spectra):
    mzs, its, precs = [], [], []
    for prec, mz, it in spectra:
        prec = float(prec)
        mz = np.asarray(mz, dtype=np.float64)
        it = np.asarray(it, dtype=np.float64)
        m = (it >= INT_FLOOR) & (mz <= prec + 2.0) & (mz > 0)
        if not np.any(m):
            continue
        mzs.append(mz[m])
        its.append(it[m])
        precs.append(prec)
    if not mzs:
        return None
    mz = np.concatenate(mzs)
    it = np.concatenate(its)
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    # dedup within 10 ppm, keep max intensity
    keep_mz, keep_it = [], []
    i = 0
    while i < len(mz):
        j = i
        tol = mz[i] * 10e-6
        while j + 1 < len(mz) and mz[j + 1] - mz[i] <= tol:
            j += 1
        k = i + int(np.argmax(it[i:j + 1]))
        keep_mz.append(mz[k])
        keep_it.append(it[k])
        i = j + 1
    mz = np.array(keep_mz)
    it = np.array(keep_it)
    if len(mz) > MAX_QPEAKS:
        k = np.argsort(-it)[:MAX_QPEAKS]
        o2 = np.argsort(mz[k])
        mz, it = mz[k][o2], it[k][o2]
    return float(np.median(precs)), mz, it


def main():
    print("Loading raw-peak index...", flush=True)
    idx = np.load(INDEX_PATH)
    precs = idx['precursors'].astype(np.float64)
    smiles = [str(s) for s in idx['smiles']]
    starts, lens = idx['p_starts'], idx['p_lens']
    flat_mz = idx['flat_mz'].astype(np.float64)
    flat_int = idx['flat_int'].astype(np.float64)
    flat_rung = (np.log(flat_mz) / LN_PHI).astype(np.float64)  # precomputed once
    n_ref = len(precs)
    print(f"Index: {n_ref} spectra", flush=True)

    def ref_peaks(i):
        s = int(starts[i])
        e = s + int(lens[i])
        return flat_mz[s:e], flat_int[s:e], flat_rung[s:e]

    print("Loading test data...", flush=True)
    test_df = pd.read_parquet(os.path.join(INPUT_DIR, 'test.parquet'))
    mol_spectra = defaultdict(list)
    for _, row in test_df.iterrows():
        mol_spectra[str(row['molecule_id'])].append(
            (row['precursor_mz'], row['ms2_mzs'], row['ms2_normalized_intensities']))
    print(f"Test: {len(test_df)} spectra, {len(mol_spectra)} molecules", flush=True)

    common_smi = Counter(smiles).most_common(1)[0][0]
    results = []
    lock_stats = defaultdict(int)

    for mi, (mol_id, spectra) in enumerate(mol_spectra.items()):
        if mi % 100 == 0:
            print(f"  {mi}/{len(mol_spectra)}...", flush=True)
        q = merge_query(spectra)
        if q is None:
            results.append({'molecule_id': mol_id,
                            'smiles': ';'.join([common_smi] * TOP_K)})
            continue
        q_prec, q_mz, q_it = q
        q_ord = np.argsort(-q_it)
        q_mz_o, q_it_o = q_mz[q_ord], q_it[q_ord]

        cand_scores = defaultdict(float)
        v1 = defaultdict(float)
        v2 = defaultdict(float)
        v3 = defaultdict(float)

        for window_tol in (PRECURSOR_TOL, 2.0):
            lo = int(np.searchsorted(precs, q_prec - window_tol, side='left'))
            hi = int(np.searchsorted(precs, q_prec + window_tol, side='right'))
            if hi > lo:
                break
        if hi <= lo:  # empty even at +-2: nearest 200 by precursor
            c = int(np.searchsorted(precs, q_prec))
            lo, hi = max(0, c - 100), min(n_ref, c + 100)

        q_r = np.log(q_mz) / LN_PHI
        q_vec1 = map1_binned(q_mz, q_it, q_prec) if USE_LOCK else None

        for ri in range(lo, hi):
            r_prec = float(precs[ri])
            r_mz, r_it, r_r = ref_peaks(ri)
            if len(r_mz) == 0:
                continue
            r_ord = np.argsort(-r_it)
            s = exact_mass_sim(q_prec, q_mz_o, q_it_o, r_prec,
                               r_mz[r_ord], r_it[r_ord])
            smi = smiles[ri]
            if s > cand_scores[smi]:
                cand_scores[smi] = s
            if USE_LOCK and s > 0:
                if q_vec1 is not None:
                    b = map1_binned(r_mz, r_it, r_prec)
                    s1 = float(q_vec1 @ b)
                    if s1 > v1[smi]:
                        v1[smi] = s1
                s2 = rung_match(q_r, q_it, r_r, r_it)
                if s2 > v2[smi]:
                    v2[smi] = s2
                s3 = interval_multiset(q_r, q_it, r_r, r_it)
                if s3 > v3[smi]:
                    v3[smi] = s3

        ranked = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        top25 = [s for s, _ in ranked[:TOP_K]]

        tier = 'BASE'
        if USE_LOCK and top25:
            votes = []
            for vd in (v1, v2, v3):
                votes.append(max(vd, key=vd.get) if vd else None)
            uniq = [v for v in votes if v]
            if len(uniq) >= 2:
                top_vote = max(set(uniq), key=uniq.count)
                n_agree = uniq.count(top_vote)
                tier = 'LOCKED' if n_agree == 3 else 'MARGINAL' if n_agree == 2 else 'SPLIT'
                if n_agree >= 2 and top_vote != top25[0] and top_vote in cand_scores:
                    top25 = [top_vote] + [s for s in top25 if s != top_vote]
            else:
                tier = 'REJECTED'
            lock_stats[tier] += 1

        if not top25:
            top25 = [common_smi] * TOP_K
        while len(top25) < TOP_K:
            top25.append(top25[0])
        results.append({'molecule_id': mol_id, 'smiles': ';'.join(top25[:TOP_K])})

    out_path = OUT_PATH
    with open(out_path, 'w', newline='\n') as f:
        f.write('molecule_id,smiles\n')
        for r in results:
            f.write(f"{r['molecule_id']},{r['smiles'].replace(chr(10), '')}\n")
    with open(out_path, 'rb') as f:
        assert b'\r' not in f.read(), "CR bytes found!"
    print(f"Wrote {len(results)} predictions -> {out_path}", flush=True)
    if USE_LOCK:
        print("Lock tiers:", dict(lock_stats), flush=True)


if __name__ == '__main__':
    main()
