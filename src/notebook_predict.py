"""
CASMI 2026 Rung 1 - Runtime Prediction Notebook Code
Reads test.parquet at run time, generates predictions via spectral library search.
Designed to work with both visible and hidden test sets.
"""

import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict

# Paths (Kaggle notebook environment)
INPUT_DIR = '/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra'
DATASET_DIR = '/kaggle/input/casmi26-baseline-index'  # Uploaded dataset with search index

N_BINS = 1500
PRECURSOR_TOL = 0.5
SIM_THRESHOLD = 0.2
TOP_K = 20

def spectrum_to_vector(mzs, intensities, precursor_mz):
    """Convert spectrum to L2-normalized binned vector."""
    vec = np.zeros(N_BINS, dtype=np.float32)
    mzs = np.asarray(mzs)
    intensities = np.asarray(intensities)
    mask = (intensities >= 0.01) & (mzs <= precursor_mz + 2.0) & (mzs < N_BINS)
    mzs_f = mzs[mask]
    int_f = intensities[mask]
    bins = np.clip((mzs_f / 1.0).astype(int), 0, N_BINS - 1)
    # Use bincount for speed
    vec += np.bincount(bins, weights=int_f, minlength=N_BINS)
    vec = np.sqrt(vec)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

def main():
    print("Loading search index...")
    precursors = np.load(os.path.join(DATASET_DIR, 'search_precursors.npy'))
    vectors_f16 = np.load(os.path.join(DATASET_DIR, 'search_vectors_f16.npy'))
    with open(os.path.join(DATASET_DIR, 'search_smiles.pkl'), 'rb') as f:
        smiles_list = pickle.load(f)
    
    # Convert to float32 for computation
    vectors = vectors_f16.astype(np.float32)
    del vectors_f16
    
    print(f"Index: {len(precursors)} vectors")
    
    print("Loading test data...")
    test_df = pd.read_parquet(os.path.join(INPUT_DIR, 'test.parquet'))
    print(f"Test: {len(test_df)} spectra, {test_df['molecule_id'].nunique()} molecules")
    
    # Group spectra by molecule
    mol_spectra = defaultdict(list)
    for idx, row in test_df.iterrows():
        mol_spectra[row['molecule_id']].append(row)
    
    print(f"Generating predictions for {len(mol_spectra)} molecules...")
    
    results = []
    for mol_id, spectra in mol_spectra.items():
        # Aggregate candidates across all spectra for this molecule
        cand_scores = defaultdict(float)
        
        for spec in spectra:
            prec = float(spec['precursor_mz'])
            try:
                qvec = spectrum_to_vector(
                    spec['ms2_mzs'],
                    spec['ms2_normalized_intensities'],
                    prec
                )
            except:
                continue
            
            if np.linalg.norm(qvec) == 0:
                continue
            
            # Binary search for precursor window
            lo = np.searchsorted(precursors, prec - PRECURSOR_TOL, side='left')
            hi = np.searchsorted(precursors, prec + PRECURSOR_TOL, side='right')
            
            if hi <= lo:
                continue
            
            # Cosine similarity via dot product (vectors are L2-normalized)
            cand_vecs = vectors[lo:hi]
            sims = cand_vecs @ qvec
            
            # Top-K above threshold
            top_idx = np.argsort(sims)[::-1][:TOP_K]
            for idx in top_idx:
                sim = float(sims[idx])
                if sim < SIM_THRESHOLD:
                    break
                smi = smiles_list[lo + idx]
                # Aggregate: sum of similarities
                cand_scores[smi] += sim
        
        # Rank by aggregated score, take top 25
        ranked = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        top25 = [smi for smi, _ in ranked[:25]]
        
        # If no candidates, use empty (should not happen, but safe)
        # Actually, to avoid format errors, if empty, we need SOMETHING.
        # Use the closest precursor match regardless of threshold as fallback.
        if not top25:
            # Fallback: find single best match with no threshold
            best_smi = None
            best_sim = -1
            for spec in spectra:
                prec = float(spec['precursor_mz'])
                try:
                    qvec = spectrum_to_vector(spec['ms2_mzs'], spec['ms2_normalized_intensities'], prec)
                except:
                    continue
                lo = np.searchsorted(precursors, prec - PRECURSOR_TOL, side='left')
                hi = np.searchsorted(precursors, prec + PRECURSOR_TOL, side='right')
                if hi <= lo:
                    continue
                sims = vectors[lo:hi] @ qvec
                bi = np.argmax(sims)
                if float(sims[bi]) > best_sim:
                    best_sim = float(sims[bi])
                    best_smi = smiles_list[lo + bi]
            if best_smi:
                top25 = [best_smi] * 25
            else:
                # Absolute last resort: use most common SMILES in index
                from collections import Counter
                most_common = Counter(smiles_list).most_common(1)[0][0]
                top25 = [most_common] * 25
        
        # Pad to 25 if needed
        while len(top25) < 25:
            top25.append(top25[0])
        
        results.append({
            'molecule_id': mol_id,
            'smiles': ';'.join(top25[:25])
        })
    
    # Write submission.csv with LF line endings
    out_df = pd.DataFrame(results)
    # Ensure molecule_id order doesn't matter, but keep as generated
    out_path = '/kaggle/working/submission.csv'
    with open(out_path, 'w', newline='\n') as f:
        f.write('molecule_id,smiles\n')
        for _, row in out_df.iterrows():
            f.write(f"{row['molecule_id']},{row['smiles']}\n")
    
    print(f"Wrote {len(results)} predictions to {out_path}")
    
    # Verify
    verify_df = pd.read_csv(out_path)
    assert list(verify_df.columns) == ['molecule_id', 'smiles']
    assert len(verify_df) == len(mol_spectra)
    assert verify_df['smiles'].str.len().min() > 0, "Empty SMILES found!"
    print("Verification passed!")

if __name__ == '__main__':
    main()
