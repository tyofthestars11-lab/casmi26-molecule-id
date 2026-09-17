"""Baseline: spectral similarity search + database retrieval.
Class 1 (in public libraries): cosine similarity against training spectra.
Class 2 (known structure, no spectra): formula -> PubChem/COCONUT lookup.
Class 3 (novel): de novo via transformer (Mass2SMILES-style)."""
import numpy as np
from collections import defaultdict

def cosine_similarity(spec1_mzs, spec1_int, spec2_mzs, spec2_int, tol=0.01):
    """Cosine similarity between two spectra with m/z tolerance."""
    # Bin matching: for each peak in spec1, find closest in spec2 within tol
    scores = []
    for mz1, int1 in zip(spec1_mzs, spec1_int):
        diffs = np.abs(spec2_mzs - mz1)
        best = np.argmin(diffs)
        if diffs[best] <= tol:
            scores.append(int1 * spec2_int[best])
    if not scores:
        return 0.0
    norm1 = np.sqrt(np.sum(np.array(spec1_int)**2))
    norm2 = np.sqrt(np.sum(np.array(spec2_int)**2))
    return sum(scores) / (norm1 * norm2 + 1e-9)

def aggregate_molecule_predictions(spectra_predictions):
    """Combine per-spectrum predictions into per-molecule ranked list (up to 25)."""
    # Weight by base_peak_intensity (higher = less noise)
    # Return top 25 unique SMILES ranked by aggregated score
    combined = defaultdict(float)
    for smiles, score, weight in spectra_predictions:
        combined[smiles] += score * weight
    ranked = sorted(combined.items(), key=lambda x: -x[1])
    return [s for s, _ in ranked[:25]]
