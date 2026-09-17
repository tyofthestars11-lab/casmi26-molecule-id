"""
CASMI 2026 - Runtime Prediction Notebook
Reads test.parquet at run time, generates predictions via spectral library search.
Works with both visible and hidden test sets.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os

# Paths
INPUT_DIR = '/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra'
INDEX_PATH = '/kaggle/input/casmi26-baseline-index/casmi_index.npz'

N_BINS = 1500
PRECURSOR_TOL = 0.5
SIM_THRESHOLD = 0.2
TOP_K = 20

def spectrum_to_vector(mzs, intensities, precursor_mz):
    vec = np.zeros(N_BINS, dtype=np.float32)
    mzs = np.asarray(mzs)
    intensities = np.asarray(intensities)
    mask = (intensities >= 0.01) & (mzs <= precursor_mz + 2.0) & (mzs < N_BINS)
    mzs_f = mzs[mask]
    int_f = intensities[mask]
    bins = np.clip(mzs_f.astype(int), 0, N_BINS - 1)
    for b, w in zip(bins, int_f):
        vec[b] += w
    vec = np.sqrt(vec)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

def main():
    print("Loading search index...")
    idx = np.load(INDEX_PATH)
    precursors = idx['precursors']
    vectors = idx['vectors'].astype(np.float32)  # f16 -> f32 for compute
    smiles_list = [str(s) for s in idx['smiles']]
    print(f"Index: {len(precursors)} vectors")

    print("Loading test data...")
    test_df = pd.read_parquet(os.path.join(INPUT_DIR, 'test.parquet'))
    print(f"Test: {len(test_df)} spectra, {test_df['molecule_id'].nunique()} molecules")

    mol_spectra = defaultdict(list)
    for _, row in test_df.iterrows():
        mol_spectra[row['molecule_id']].append(row)

    print(f"Predicting for {len(mol_spectra)} molecules...")
    results = []
    for i, (mol_id, spectra) in enumerate(mol_spectra.items()):
        if i % 100 == 0:
            print(f"  {i}/{len(mol_spectra)}...")
        cand_scores = defaultdict(float)
        for spec in spectra:
            prec = float(spec['precursor_mz'])
            try:
                qvec = spectrum_to_vector(spec['ms2_mzs'], spec['ms2_normalized_intensities'], prec)
            except:
                continue
            if np.linalg.norm(qvec) == 0:
                continue
            lo = np.searchsorted(precursors, prec - PRECURSOR_TOL, side='left')
            hi = np.searchsorted(precursors, prec + PRECURSOR_TOL, side='right')
            if hi <= lo:
                continue
            sims = vectors[lo:hi] @ qvec
            top_idx = np.argsort(sims)[::-1][:TOP_K]
            for j in top_idx:
                sim = float(sims[j])
                if sim < SIM_THRESHOLD:
                    break
                cand_scores[smiles_list[lo + j]] += sim
        
        ranked = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        top25 = [smi for smi, _ in ranked[:25]]
        
        if not top25:
            # Fallback: best match with no threshold
            best_smi, best_sim = None, -1
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
                bi = int(np.argmax(sims))
                if float(sims[bi]) > best_sim:
                    best_sim = float(sims[bi])
                    best_smi = smiles_list[lo + bi]
            if best_smi:
                top25 = [best_smi] * 25
            else:
                # Last resort: most common SMILES
                from collections import Counter
                top25 = [Counter(smiles_list).most_common(1)[0][0]] * 25
        
        while len(top25) < 25:
            top25.append(top25[0])
        results.append({'molecule_id': mol_id, 'smiles': ';'.join(top25[:25])})

    # Write with LF-only line endings
    out_path = '/kaggle/working/submission.csv'
    with open(out_path, 'w', newline='\n') as f:
        f.write('molecule_id,smiles\n')
        for r in results:
            # Sanitize: no newlines or commas in SMILES (shouldn't happen, but safe)
            smi_clean = r['smiles'].replace('\n', '').replace('\r', '')
            f.write(f"{r['molecule_id']},{smi_clean}\n")
    
    print(f"Wrote {len(results)} predictions")
    
    # Verify no CR bytes
    with open(out_path, 'rb') as f:
        data = f.read()
    assert b'\r' not in data, "CR bytes found!"
    print("LF-only verified, no CR bytes")

if __name__ == '__main__':
    main()
