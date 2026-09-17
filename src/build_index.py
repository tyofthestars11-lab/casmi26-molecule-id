"""
CASMI 2026 - Rung 1 search index builder.

Streams train.parquet, vectorizes spectra from the Enveda libraries
(enveda-np-examples, enveda-180), caps at MAX_VECTORS via memory-mapped
arrays so the build fits in a small environment, sorts by precursor m/z
for fast binary-search windowing, and saves the index.

Output: models/search_precursors.npy, models/search_vectors_f16.npy,
        models/search_smiles.pkl  (packed to models/casmi_index.npz
        for the Kaggle dataset upload)

Vector recipe (must match src/notebook_predict_v2.py):
  - 1 Da bins, 1500 dimensions
  - relative intensity floor 0.01
  - peaks above precursor + 2 Da excluded
  - square-root intensity transform, L2 normalized
"""

import gc
import os

import numpy as np
import pyarrow.parquet as pq
import pickle

N_BINS = 1500
MAX_VECTORS = 150000  # cap to fit small build environments


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


def main(train_path="data/train.parquet", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    print(f"Building index (max {MAX_VECTORS} vectors)...", flush=True)

    # Pre-allocate memory-mapped arrays so the peak RSS stays small.
    precursors_mm = np.memmap(
        os.path.join(models_dir, "precursors_tmp.dat"),
        dtype=np.float32, mode="w+", shape=(MAX_VECTORS,))
    vectors_mm = np.memmap(
        os.path.join(models_dir, "vectors_tmp.dat"),
        dtype=np.float16, mode="w+", shape=(MAX_VECTORS, N_BINS))
    smiles_list = []

    pf = pq.ParquetFile(train_path)
    count = 0
    batch_num = 0
    for batch in pf.iter_batches(
            batch_size=100000,
            columns=["ms2_mzs", "ms2_normalized_intensities",
                     "precursor_mz", "normalized_smiles", "ingest_lib"]):
        if count >= MAX_VECTORS:
            break
        batch_num += 1
        df = batch.to_pandas()
        df = df[df["ingest_lib"].isin(["enveda-np-examples", "enveda-180"])]
        print(f"Batch {batch_num}: {len(df)} rows, count: {count}", flush=True)
        for _, row in df.iterrows():
            if count >= MAX_VECTORS:
                break
            try:
                v = spectrum_to_vector(row["ms2_mzs"],
                                       row["ms2_normalized_intensities"],
                                       float(row["precursor_mz"]))
            except Exception:
                continue
            if np.linalg.norm(v) > 0:
                precursors_mm[count] = float(row["precursor_mz"])
                vectors_mm[count] = v.astype(np.float16)
                smiles_list.append(str(row["normalized_smiles"]))
                count += 1
        del df
        gc.collect()

    print(f"\nCollected {count} vectors", flush=True)

    precursors = np.array(precursors_mm[:count])
    vectors_f16 = np.array(vectors_mm[:count])
    smiles_list = smiles_list[:count]

    # Sort by precursor for binary-search windowing at predict time.
    sort_idx = np.argsort(precursors)
    precursors = precursors[sort_idx]
    vectors_f16 = vectors_f16[sort_idx]
    smiles_list = [smiles_list[i] for i in sort_idx]
    print(f"Sorted. Vectors: {vectors_f16.shape}, "
          f"{vectors_f16.nbytes / 1e6:.1f} MB", flush=True)

    np.save(os.path.join(models_dir, "search_precursors.npy"), precursors)
    np.save(os.path.join(models_dir, "search_vectors_f16.npy"), vectors_f16)
    with open(os.path.join(models_dir, "search_smiles.pkl"), "wb") as f:
        pickle.dump(smiles_list, f)

    # Compressed transfer artifact uploaded to Kaggle as the index dataset.
    np.savez_compressed(
        os.path.join(models_dir, "casmi_index.npz"),
        precursors=precursors, vectors=vectors_f16,
        smiles=np.array(smiles_list))

    os.remove(os.path.join(models_dir, "precursors_tmp.dat"))
    os.remove(os.path.join(models_dir, "vectors_tmp.dat"))
    print("Index saved!", flush=True)


if __name__ == "__main__":
    main()
