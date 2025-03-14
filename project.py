from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
from nomad_projection import NomadProjection

if __name__ == '__main__':

    # Load and stack all .npy files from the preproc output directory
    data_dir = Path('./nomic-embed-v2-wikivecs/')
    npy_files = sorted(data_dir.glob('*/*.npy'))

    if not npy_files:
        raise FileNotFoundError("No .npy files found in the data directory.")

    # Check if "x.npy" exists in the data directory
    x_file = data_dir / 'merged.npy'
    if x_file.exists():
        print("Loading pre-stacked data ")
        s = time.time()
        x = np.load(x_file)
        e = time.time()
        print(f'Loading took: {e - s:.2f} seconds')
    else:
        print("merged.npy not found. Loading and combining individual embedding files.")
        # Preallocate the array
        embedding_files = [f for f in npy_files if 'embeddings' in f.name]
        # Sort the files alphabetically
        embedding_files.sort()

        x = np.empty((61614907, 768))
        start_idx = 0
        for npy_file in tqdm(npy_files):
            current_data = np.load(npy_file)
            end_idx = start_idx + current_data.shape[0]
            x[start_idx:end_idx] = current_data
            start_idx = end_idx

        np.save(x_file, x)
        print(f"Saved combined embeddings to {x_file}")

    print(f"Final loaded data shape: {x.shape}")

    p = NomadProjection()
    low_d = p.fit_transform(x,
                    n_noise=10_000,
                    n_neighbors=64,
                    n_cells=128,
                    epochs=600,
                    momentum=0.0,
                    lr_scale=0.1,
                    learning_rate_decay_start_time=0.1,
                    late_exaggeration_time=0.7,
                    late_exaggeration_scale=1.5,
                    batch_size=70_000,
                    cluster_subset_size=2500000,
                    cluster_chunk_size=1000,
                    debug_plot=False,
    )

    print('Saving projection')
    with open('./low_d.npy', 'wb') as f:
        np.save(f, low_d)