"""
spectral_lock4way.py — lock4way ported to spectra.

The operator: four independent maps read the same data; LOCKED iff the
peak agrees in ALL FOUR maps (2 seeds x 2 directions -> 4 witnesses);
MARGINAL iff 3 of 4; else REJECTED.

On spectra, the four maps are four independent reads of the same peak list:
  Map 1 — raw intensity baseline: 1-Da binned cosine (Rung 1 method)
  Map 2 — phi-spaced rung quantization: tiled rung-histogram cosine
  Map 3 — forward/reverse conjugate: rung-interval sequence match,
          averaged over the forward and phase-conjugate reads
  Map 4 — reflection invariant: even component of the rung signature
          about the precursor rung, cosine matched

Each map votes its top-1 molecule. The lock verdict follows the same rule
as the scroll-ink lock: unanimity or nothing.
"""
import numpy as np
from spectral_tiled_lucas import (
    clean_peaks, rung_histogram, histogram_cosine, rung_address, LN_PHI)
from spectral_fc_chars import (
    rung_signature, signature_cosine, conjugate_signature,
    interval_sequence, interval_match, reflection_symmetric,
    symmetric_cosine)

N_BINS = 1500


def map1_vector(mzs, intensities, precursor_mz):
    """Raw intensity baseline: 1-Da bins, sqrt, L2."""
    vec = np.zeros(N_BINS, dtype=np.float32)
    mzs, intensities = clean_peaks(mzs, intensities, precursor_mz)
    if len(mzs):
        bins = np.clip(mzs.astype(int), 0, N_BINS - 1)
        for b, w in zip(bins, intensities):
            vec[b] += w
    vec = np.sqrt(vec)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def map1_score(q_vec, r_vec):
    return float(q_vec @ r_vec)


def map2_score(q, r):
    """Phi-spaced rung quantization read."""
    qc, qw = rung_histogram(*q)
    rc, rw = rung_histogram(*r)
    return histogram_cosine(qc, qw, rc, rw)


def map3_score(q, r):
    """Forward/reverse conjugate read: interval match averaged over
    the forward signature and its phase-conjugate twin."""
    q_iv, q_w = interval_sequence(*q)
    r_iv, r_w = interval_sequence(*r)
    fwd = interval_match(q_iv, q_w, r_iv, r_w)
    # conjugate read: same intervals, reversed order (deterministic twin)
    if len(q_iv) and len(r_iv):
        conj = interval_match(q_iv[::-1], q_w[::-1], r_iv[::-1], r_w[::-1])
    else:
        conj = 0.0
    return 0.5 * (fwd + conj)


def map4_score(q, r):
    """Reflection invariant: even component about the precursor rung."""
    q_sig = rung_signature(*q)
    r_sig = rung_signature(*r)
    qc = rung_address(np.array([q[2]])) if q[2] else np.array([0.0])
    rc = rung_address(np.array([r[2]])) if r[2] else np.array([0.0])
    q_sym = reflection_symmetric(q_sig, float(qc[0]))
    r_sym = reflection_symmetric(r_sig, float(rc[0]))
    return symmetric_cosine(q_sym, r_sym)


MAPS = {
    'map1_raw': map1_score,
    'map2_rung': map2_score,
    'map3_conjugate': map3_score,
    'map4_reflection': map4_score,
}


def lock_verdict(votes):
    """votes: 4 top-1 molecule calls. -> LOCKED / MARGINAL / REJECTED."""
    if len(set(votes)) == 1:
        return 'LOCKED'
    top, count = max(((v, votes.count(v)) for v in set(votes)),
                     key=lambda x: x[1])
    if count == 3:
        return 'MARGINAL'
    return 'REJECTED'
