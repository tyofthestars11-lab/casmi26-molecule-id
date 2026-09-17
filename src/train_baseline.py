"""Train baseline: spectral library search.
For each test molecule, find most similar training spectra by cosine similarity,
aggregate predictions per molecule, output top-25 SMILES."""
import pandas as pd
import numpy as np
from data_loader import load_train, load_test, curate_spectrum
from baseline_model import cosine_similarity, aggregate_molecule_predictions

def build_library_index(train_df, max_spectra=50000):
    """Build searchable index: curated spectra with SMILES labels.
    Subsample for speed; prioritize enveda-np-examples (closest to test)."""
    # Prioritize instrument-matched libraries
    priority = train_df[train_df['ingest_lib'].isin(['enveda-np-examples', 'enveda-180'])]
    rest = train_df[~train_df.index.isin(priority.index)]
    sample = pd.concat([priority, rest.sample(min(len(rest), max_spectra - len(priority)))])
    
    index = []
    for _, row in sample.iterrows():
        mzs, ints = curate_spectrum(row['ms2_mzs'], row['ms2_normalized_intensities'], row['precursor_mz'])
        if len(mzs) >= 5:  # Minimum informative peaks
            index.append({
                'mzs': mzs, 'ints': ints,
                'smiles': row['normalized_smiles'],
                'precursor_mz': row['precursor_mz'],
            })
    print(f"Index: {len(index)} spectra")
    return index

def predict_molecule(molecule_spectra, library_index, top_k=25):
    """Predict top-25 SMILES for one molecule from its spectra."""
    all_preds = []
    for spec in molecule_spectra:
        mzs, ints = curate_spectrum(spec['ms2_mzs'], spec['ms2_normalized_intensities'], spec['precursor_mz'])
        if len(mzs) < 5:
            continue
        # Find top matches in library (precursor filter first for speed)
        candidates = []
        for lib in library_index:
            if abs(lib['precursor_mz'] - spec['precursor_mz']) > 0.5:
                continue
            sim = cosine_similarity(mzs, ints, lib['mzs'], lib['ints'])
            if sim > 0.3:
                candidates.append((lib['smiles'], sim))
        candidates.sort(key=lambda x: -x[1])
        for smiles, sim in candidates[:10]:
            all_preds.append((smiles, sim, spec['base_peak_intensity'] or 1.0))
    return aggregate_molecule_predictions(all_preds)[:top_k]

if __name__ == '__main__':
    print("Baseline training script ready. Needs data.")
