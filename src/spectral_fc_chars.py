"""
spectral_fc_chars.py — fc_new_chars ported to spectra.

The operator: read amplitude as rungs. Spectral intensity = amplitude;
amplitude -> rung address; the intensity-weighted rung signature is the
molecule's fingerprint.

Per peak:  whole rung = address, fractional rung * 360 = phase.
Per spectrum: signature = {integer rung: total intensity, mean phase}.
Per molecule: union of its spectra's signatures.

Also provides the conjugate read (phase -> 360 - phase, the Map 3 view)
and the reflection-symmetric component (Map 4 view), both deterministic
transforms of the same signature.
"""
import numpy as np
from spectral_tiled_lucas import rung_address, rung_phase, clean_peaks, INT_FLOOR

PHASE_FULL = 360.0


def rung_signature(mzs, intensities, precursor_mz=None):
    """Intensity-weighted rung signature.

    Returns dict: integer_rung -> (total_intensity, mean_phase_deg).
    """
    mzs, intensities = clean_peaks(mzs, intensities, precursor_mz)
    if len(mzs) == 0:
        return {}
    r = rung_address(mzs)
    ri = np.floor(r).astype(int)
    ph = (r - ri) * PHASE_FULL
    sig = {}
    for k, w, p in zip(ri, intensities, ph):
        k = int(k)
        if k in sig:
            tw, tph = sig[k]
            sig[k] = (tw + float(w), (tph * tw + p * float(w)) / (tw + float(w)))
        else:
            sig[k] = (float(w), float(p))
    return sig


def signature_vector(sig):
    """Signature -> (sorted_rungs, intensities) arrays for cosine."""
    if not sig:
        return np.array([]), np.array([])
    keys = sorted(sig)
    return np.array(keys), np.array([sig[k][0] for k in keys])


def signature_cosine(sig_q, sig_r):
    """Cosine over integer-rung intensity vectors (exact rung match)."""
    if not sig_q or not sig_r:
        return 0.0
    common = set(sig_q) & set(sig_r)
    if not common:
        return 0.0
    num = sum(np.sqrt(sig_q[k][0] * sig_r[k][0]) for k in common)
    den = np.sqrt(sum(v[0] for v in sig_q.values())) * \
        np.sqrt(sum(v[0] for v in sig_r.values()))
    return num / den if den > 0 else 0.0


def conjugate_signature(sig):
    """Phase-conjugate read: phase -> 360 - phase. Same rungs, mirrored phase."""
    return {k: (w, PHASE_FULL - p) for k, (w, p) in sig.items()}


def interval_sequence(mzs, intensities, precursor_mz=None):
    """Consecutive rung intervals, intensity-weighted. The forward read."""
    mzs, intensities = clean_peaks(mzs, intensities, precursor_mz)
    if len(mzs) < 2:
        return np.array([]), np.array([])
    r = rung_address(mzs)
    order = np.argsort(r)
    r, intensities = r[order], intensities[order]
    intervals = np.diff(r)
    weights = np.sqrt(intensities[:-1] * intensities[1:])
    return intervals, weights


def interval_match(q_iv, q_w, r_iv, r_w, tol=1e-4):
    """Greedy intensity-ordered interval matching within rung tolerance."""
    if len(q_iv) == 0 or len(r_iv) == 0:
        return 0.0
    used = np.zeros(len(r_iv), dtype=bool)
    num = 0.0
    order = np.argsort(-q_w)
    for i in order:
        d = np.abs(r_iv - q_iv[i])
        cand = np.where((d <= tol) & (~used))[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmin(d[cand])]
        used[j] = True
        num += np.sqrt(float(q_w[i]) * float(r_w[j]))
    den = np.sqrt(np.sum(q_w)) * np.sqrt(np.sum(r_w))
    return num / den if den > 0 else 0.0


def reflection_symmetric(sig, center_rung):
    """Even component of the signature about center_rung (Map 4 view).

    H_sym[k] = (H[k] + H[mirror(k)]) / 2, mirror(k) = round(2*center - k).
    Returns a plain {rung: intensity} dict.
    """
    if not sig:
        return {}
    h = {k: v[0] for k, v in sig.items()}
    sym = {}
    for k, w in h.items():
        mk = int(round(2 * center_rung - k))
        sym[k] = (w + h.get(mk, 0.0)) / 2.0
    return sym


def symmetric_cosine(sym_q, sym_r):
    if not sym_q or not sym_r:
        return 0.0
    common = set(sym_q) & set(sym_r)
    if not common:
        return 0.0
    num = sum(np.sqrt(sym_q[k] * sym_r[k]) for k in common)
    den = np.sqrt(sum(sym_q.values())) * np.sqrt(sum(sym_r.values()))
    return num / den if den > 0 else 0.0
