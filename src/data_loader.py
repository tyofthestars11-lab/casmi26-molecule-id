"""CASMI 2026 data loader - MS/MS spectra to molecule structure."""
import pandas as pd
import numpy as np

# Column specs from competition page
TEST_COLS = ['molecule_id', 'spectrum_id', 'ms2_mzs', 'ms2_normalized_intensities',
             'base_peak_intensity', 'adduct', 'ionization_mode', 'instrument_type',
             'precursor_mz', 'collision_energy_ev', 'collision_energy_orig',
             'collision_energy_orig_units']
TRAIN_EXTRA = ['normalized_smiles', 'inchikey', 'inchikey14', 'molecular_formula',
               'ingest_lib', 'adduct_orig', 'precursor_error_ppm', 'num_peaks']

def load_train(path):
    """Load training spectra with SMILES labels."""
    df = pd.read_parquet(path)
    print(f"Train: {len(df)} spectra, {df['normalized_smiles'].nunique()} unique structures")
    return df

def load_test(path):
    """Load test spectra (no labels)."""
    df = pd.read_parquet(path)
    print(f"Test: {len(df)} spectra, {df['molecule_id'].nunique()} molecules")
    return df

def curate_spectrum(mzs, intensities, precursor_mz, intensity_floor=0.01, max_peaks=128):
    """Standard MS/MS curation: floor, precursor filter, top-N, sqrt transform."""
    mzs = np.array(mzs)
    intensities = np.array(intensities)
    # Drop peaks above precursor (contaminants)
    mask = mzs <= precursor_mz + 2.0
    mzs, intensities = mzs[mask], intensities[mask]
    # Intensity floor
    mask = intensities >= intensity_floor
    mzs, intensities = mzs[mask], intensities[mask]
    # Top-N by intensity
    if len(mzs) > max_peaks:
        idx = np.argsort(intensities)[-max_peaks:]
        mzs, intensities = mzs[idx], intensities[idx]
    # Sqrt transform (compress dynamic range)
    intensities = np.sqrt(intensities)
    # Sort by m/z
    order = np.argsort(mzs)
    return mzs[order], intensities[order]
