"""
Exact-mass spectral search — chemistry as fuel.

Replaces 1-Da binned cosine with ppm-tolerance peak matching:
  - fragments matched at PPM_TOL (high-res data: 4 decimals, binning wastes it)
  - neutral losses (precursor - fragment) matched as a second fingerprint
  - greedy intensity-ordered matching, no peak reused
Score: 0.7 * fragment cosine + 0.3 * neutral-loss cosine.
"""

import numpy as np

PPM_TOL = 20.0
INT_FLOOR = 0.01
FRAG_WEIGHT = 0.7
LOSS_WEIGHT = 0.3


def clean_peaks(mzs, intensities, precursor_mz=None):
    mzs = np.asarray(mzs, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    mask = intensities >= INT_FLOOR
    if precursor_mz is not None:
        mask &= mzs <= precursor_mz + 2.0
    mzs, intensities = mzs[mask], intensities[mask]
    if len(mzs) == 0:
        return mzs, intensities
    order = np.argsort(-intensities)
    return mzs[order], intensities[order]


def loss_spectrum(precursor_mz, mzs, intensities):
    """Neutral losses as (loss_mz, intensity), intensity-ordered."""
    losses = precursor_mz - mzs
    mask = losses > 0
    return losses[mask], intensities[mask]


def _match(q_mz, q_int, r_mz, r_int, ppm):
    """Greedy intensity-ordered peak matching. Returns sum of sqrt(Iq*Ir)."""
    if len(q_mz) == 0 or len(r_mz) == 0:
        return 0.0
    used = np.zeros(len(r_mz), dtype=bool)
    score = 0.0
    for mz, inten in zip(q_mz, q_int):
        tol = mz * ppm * 1e-6
        d = np.abs(r_mz - mz)
        cand = np.where((d <= tol) & (~used))[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmin(d[cand])]
        used[j] = True
        score += np.sqrt(float(inten) * float(r_int[j]))
    return score


def _cosine_from_match(match_sum, q_int, r_int):
    nq = np.sqrt(np.sum(q_int))
    nr = np.sqrt(np.sum(r_int))
    if nq == 0 or nr == 0:
        return 0.0
    return match_sum / (nq * nr)


def similarity(q_prec, q_mz, q_int, r_prec, r_mz, r_int, ppm=PPM_TOL):
    """Combined fragment + neutral-loss similarity in [0, 1]."""
    q_mz, q_int = clean_peaks(q_mz, q_int, q_prec)
    r_mz, r_int = clean_peaks(r_mz, r_int, r_prec)
    frag = _cosine_from_match(_match(q_mz, q_int, r_mz, r_int, ppm),
                              q_int, r_int)
    q_lm, q_li = loss_spectrum(q_prec, q_mz, q_int)
    r_lm, r_li = loss_spectrum(r_prec, r_mz, r_int)
    loss = _cosine_from_match(_match(q_lm, q_li, r_lm, r_li, ppm),
                              q_li, r_li)
    return FRAG_WEIGHT * frag + LOSS_WEIGHT * loss
