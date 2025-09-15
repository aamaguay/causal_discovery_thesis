import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import StandardScaler
from causa.loci import loci, loci_w_marginal, compute_marginal_likelihood_nn
from causa.datasets import MNU, Tuebingen, SIM, SIMc, SIMG, SIMln, Cha, Multi, Net
from causa.utils import plot_pair

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

# ls_dataset_used = [
#             ('MNU', MNU, 100),
#             ('Cha', Cha, 300),
#             ('Multi', Multi, 300),
#             ('Net', Net, 300),
#             ('SIM', SIM, 100),
#             ('SIMc', SIMc, 100),
#             ('SIMG', SIMG, 100),
#             ('SIMln', SIMln, 100),
#             ('Tuebingen', Tuebingen, 108)
#     ]

# ('MNU', MNU, 100)
# ('Cha', Cha, 300)
# ('SIM', SIM, 100)
# ('SIMc', SIMc, 100)
# ('SIMG', SIMG, 100)
# ('SIMln', SIMln, 100)
# ('Tuebingen', Tuebingen, 108)

ls_dataset_used = [
            ('Cha', Cha, 300)
    ]

ls_all_results_marginals = {'Cha': []} # {name: [] for name, _, _ in ls_dataset_used}

def estimate_sample(ds_class, idx, n_steps, seed=711, device='cpu', preprocessor = None):
    try:
        print(f"start estimation, sample {idx}")
        dataset = ds_class(idx, preprocessor=preprocessor, double=True)
        x = dataset.cause.flatten().numpy()
        y = dataset.effect.flatten().numpy()

        log_mg_x, _, _ = compute_marginal_likelihood_nn(x, n_steps = n_steps, seed=seed, device=device)
        log_mg_y, _, _ = compute_marginal_likelihood_nn(y, n_steps = n_steps, seed=seed, device=device)

        marginal_diff = log_mg_x - log_mg_y
        print(f"successful estimation, sample {idx}")
        return idx, marginal_diff
    except Exception as e:
        print(f"[WARNING] Sample {idx} failed with error: {e}")
        return idx, np.nan


if __name__ == "__main__":
    # -------------------------------------------
    # Main loop: over datasets
    # -------------------------------------------
    max_workers = 5  # you can increase this depending on CPU
    for ds_name, ds_class, n_samples in ls_dataset_used:
        print(f"⏳ Estimating dataset: {ds_name} with {n_samples} samples...***")

        start_time = time.time()
        results = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    estimate_sample, 
                    ds_class = ds_class, idx = idx, 
                    n_steps=1000, preprocessor=StandardScaler()
                    ): idx for idx in range(1, n_samples+1)
                }
            for future in tqdm(as_completed(futures), total=n_samples, desc=f"{ds_name}"):
                idx, marginal_diff = future.result()
                results.append((idx, marginal_diff))

        # Sort by sample index to keep consistent order
        ls_all_results_marginals[ds_name] = sorted(results, key=lambda x: x[0])

        # Save to partial_results folder
        df_results = pd.DataFrame(ls_all_results_marginals[ds_name], columns=["sample_idx", "marginal_diff"])
        df_results.to_csv(f'partial_results/t1000_marginals_{ds_name}.csv', index=False)
        print(f"✅ Saved results for {ds_name} to 'partial_results/marginals_{ds_name}.csv'")

        # ⏱️ Print total duration
        end_time = time.time()
        duration_minutes = (end_time - start_time) / 60
        print(f"⏱️ Estimation for {ds_name} completed in {duration_minutes:.2f} minutes.")


