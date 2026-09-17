"""
spectral_tiled_lucas.py — tiled_lucas ported to spectra.

The operator: tile the data into φ-spaced bins, compute the centroid per
tile, map each centroid to its rung via rung = ln(x) / ln(phi).

On spectra: the data is a peak list. Tiles are φ-spaced intervals of the
m/z axis (uniform steps in rung space). Per tile: intensity-weighted
centroid -> rung address + phase. The spectrum becomes a rung sequence.
"""
import numpy as np

PHI = (1 + 5 ** 0.5) / 2
LN_PHI = np.log(PHI)
INT_FLOOR = 0.01
TILE_STEP = 0.05  # rung units per tile; 20 tiles per phi-fold


def rung_address(mz):
    """Rung address of a peak position. Vectorized."""
    return np.log(np.asarray(mz, dtype=np.float64)) / LN_PHI


def rung_phase(mz):
    """Fractional rung -> phase in [0, 1). Vectorized."""
    r = rung_address(mz)
    return r - np.floor(r)


def clean_peaks(mzs, intensities, precursor_mz=None):
    mzs = np.asarray(mzs, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    mask = intensities >= INT_FLOOR
    if precursor_mz is not None:
        mask &= mzs <= precursor_mz + 2.0
    return mzs[mask], intensities[mask]


def phi_tiles(mzs, intensities, precursor_mz=None, step=TILE_STEP):
    """Tile peaks into phi-spaced bins; intensity-weighted centroid per tile.

    Returns (tile_rung_centers, tile_intensities): the centroid of each
    non-empty tile mapped to its rung address, with summed intensity.
    """
    mzs, intensities = clean_peaks(mzs, intensities, precursor_mz)
    if len(mzs) == 0:
        return np.array([]), np.array([])
    r = rung_address(mzs)
    lo = np.floor(r.min() / step) * step
    tile_idx = np.floor((r - lo) / step).astype(int)
    n_tiles = int(tile_idx.max()) + 1
    centers = np.zeros(n_tiles)
    weights = np.zeros(n_tiles)
    for t in range(n_tiles):
        m = tile_idx == t
        if not m.any():
            continue
        w = intensities[m]
        centers[t] = np.sum(r[m] * w) / np.sum(w)  # centroid rung
        weights[t] = np.sum(w)
    keep = weights > 0
    return centers[keep], weights[keep]


def rung_histogram(mzs, intensities, precursor_mz=None, step=TILE_STEP):
    """Intensity histogram over phi-spaced tiles: the tiled spectrum."""
    centers, weights = phi_tiles(mzs, intensities, precursor_mz, step)
    return centers, weights


def histogram_cosine(qc, qw, rc, rw, tol=0.5):
    """Cosine between two tiled spectra; tiles match within tol (rung units)."""
    if len(qc) == 0 or len(rc) == 0:
        return 0.0
    used = np.zeros(len(rc), dtype=bool)
    num = 0.0
    order = np.argsort(-qw)
    for i in order:
        d = np.abs(rc - qc[i])
        cand = np.where((d <= tol) & (~used))[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmin(d[cand])]
        used[j] = True
        num += np.sqrt(float(qw[i]) * float(rw[j]))
    den = np.sqrt(np.sum(qw)) * np.sqrt(np.sum(rw))
    return num / den if den > 0 else 0.0
