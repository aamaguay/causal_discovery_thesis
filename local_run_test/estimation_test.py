# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from sklearn.preprocessing import StandardScaler
# from causa.loci import loci, loci_w_marginal, compute_marginal_likelihood_nn
# from causa.datasets import MNU, Tuebingen, SIM, SIMc, SIMG, SIMln, Cha, Multi, Net
# from causa.utils import plot_pair

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm import tqdm
# import time

# from pathlib import Path

# # Define base and results directories
# BASE_DIR = Path(__file__).resolve().parents[1]   # points to /work/smalamag/tmp/loci
# RESULTS_DIR = BASE_DIR / "final_results"
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# # ls_dataset_used = [
# #             ('MNU', MNU, 100),
# #             ('Cha', Cha, 300),
# #             ('Multi', Multi, 300),
# #             ('Net', Net, 300),
# #             ('SIM', SIM, 100),
# #             ('SIMc', SIMc, 100),
# #             ('SIMG', SIMG, 100),
# #             ('SIMln', SIMln, 100),
# #             ('Tuebingen', Tuebingen, 108)
# #     ]

# # ('MNU', MNU, 100)
# # ('Cha', Cha, 300)
# # ('SIM', SIM, 100)
# # ('SIMc', SIMc, 100)
# # ('SIMG', SIMG, 100)
# # ('SIMln', SIMln, 100)
# # ('Tuebingen', Tuebingen, 108)

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

# ls_all_results_marginals = {'MNU': [], 'Cha': [], 'Multi': [], 'Net': [], 'SIM': [], 'SIMc': [],
#                             'SIMG': [], 'SIMln': [], 'Tuebingen': [] } # {name: [] for name, _, _ in ls_dataset_used}
# ls_time_results_marginals = {'name': [], 'time': []} 

# def estimate_sample(ds_class, idx, n_steps, seed=711, device='cpu', preprocessor = None):
#     try:
#         print(f"start estimation, sample {idx}")
#         dataset = ds_class(idx, preprocessor=preprocessor, double=True)
#         x = dataset.cause.flatten().numpy()
#         y = dataset.effect.flatten().numpy()

#         log_mg_x, _, _ = compute_marginal_likelihood_nn(x, n_steps = n_steps, seed=seed, device=device)
#         log_mg_y, _, _ = compute_marginal_likelihood_nn(y, n_steps = n_steps, seed=seed, device=device)

#         marginal_diff = log_mg_x - log_mg_y
#         print(f"successful estimation, sample {idx}")
#         return idx, marginal_diff
#     except Exception as e:
#         print(f"[WARNING] Sample {idx} failed with error: {e}")
#         return idx, np.nan


# if __name__ == "__main__":
#     # -------------------------------------------
#     # Main loop: over datasets
#     # -------------------------------------------
#     max_workers = 10  # you can increase this depending on CPU
#     for ds_name, ds_class, n_samples in ls_dataset_used:
#         print(f"⏳ Estimating dataset: {ds_name} with {n_samples} samples...***")

#         start_time = time.time()
#         results = []

#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             futures = {
#                 executor.submit(
#                     estimate_sample, 
#                     ds_class = ds_class, idx = idx, 
#                     n_steps=15, preprocessor=StandardScaler()
#                     ): idx for idx in range(1, n_samples+1)[:2]
#                 }
#             for future in tqdm(as_completed(futures), total=n_samples, desc=f"{ds_name}"):
#                 idx, marginal_diff = future.result()
#                 results.append((idx, marginal_diff))

#         # Sort by sample index to keep consistent order
#         ls_all_results_marginals[ds_name] = sorted(results, key=lambda x: x[0])

#         # Save to partial_results folder
#         df_results = pd.DataFrame(ls_all_results_marginals[ds_name], columns=["sample_idx", "marginal_diff"])
#         # df_results.to_csv(f'final_results/s15_marginals_{ds_name}.csv', index=False)
#         df_results.to_csv(RESULTS_DIR / f"s15_marginals_{ds_name}.csv", index=False)
#         print(f"✅ Saved results for {ds_name} to 'final_results/s15_marginals_{ds_name}.csv'")

#         # ⏱️ Print total duration
#         end_time = time.time()
#         duration_minutes = (end_time - start_time) / 60
#         print(f"⏱️ Estimation for {ds_name} completed in {duration_minutes:.2f} minutes.")

#         ls_time_results_marginals['name'].append(ds_name)
#         ls_time_results_marginals['time'].append(duration_minutes)
    
#     df_time = pd.DataFrame(ls_time_results_marginals)
#     # df_time.to_csv(f'final_results/s15_marginals_time.csv', index=False)
#     df_time.to_csv(RESULTS_DIR / "s15_marginals_time.csv", index=False)



import os
from pathlib import Path
import argparse

from sklearn.preprocessing import StandardScaler
from causa.loci import loci, loci_w_marginal, compute_marginal_likelihood_nn
from causa.datasets import MNU, Tuebingen, SIM, SIMc, SIMG, SIMln, Cha, Multi, Net, AN, ANs, LSs, LS
from causa.utils import plot_pair

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

# ---------- CLI / ENV controls ----------
def positive_int(v):
    v = int(v)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=positive_int,
                    default=int(os.getenv("LOCI_STEPS", "2000")),
                    help="n_steps for compute_marginal_likelihood_nn")
parser.add_argument("--only", type=str, default=os.getenv("LOCI_DATASET", ""),
                    help="comma-separated dataset names to run (e.g., 'MNU,Cha')")

parser.add_argument(
    "--conf",
    type=str,
    default=os.getenv("LOCI_CONF", "conf1"),
    help="flow configuration name (e.g. conf1, conf2, conf3)"
)

parser.add_argument(
    "--flow",
    type=str,
    default=os.getenv("LOCI_FLOW", "nsf"),
    help="flow type to use (e.g. nsf, realnvp)"
)
args = parser.parse_args()



# How many workers? match Slurm CPUs (fallback to 4)
SLURM_CPUS = int(os.getenv("SLURM_CPUS_PER_TASK", "4"))
MAX_WORKERS = max(1, min(SLURM_CPUS, os.cpu_count() or SLURM_CPUS))

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[1]   # /work/smalamag/tmp/loci
RESULTS_DIR = BASE_DIR / "final_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_DATASETS = [
    ('MNU', MNU, 100),
    ('Cha', Cha, 300),
    ('Multi', Multi, 300),
    ('Net', Net, 300),
    ('SIM', SIM, 100),
    ('SIMc', SIMc, 100),
    ('SIMG', SIMG, 100),
    ('SIMln', SIMln, 100),
    ('Tuebingen', Tuebingen, 108),
    ('AN', AN, 100),
    ('ANs', ANs, 100),
    ('LS', LS, 100),
    ('LSs', LSs, 100)
]

# Optional filter
if args.only:
    wanted = {x.strip() for x in args.only.split(",")}
    DATASETS = [d for d in ALL_DATASETS if d[0] in wanted]
else:
    DATASETS = ALL_DATASETS

ls_all_results_marginals = {name: [] for name,_,_ in ALL_DATASETS}
ls_time_results_marginals = {'name': [], 'time': []}


def estimate_sample(ds_class, idx, n_steps, conf_name, flow_name, seed=711, device='cpu', preprocessor = None):
    try:
        print(f"start estimation, sample {idx}")
        dataset = ds_class(idx, preprocessor=preprocessor, double=True)
        x = dataset.cause.flatten().numpy()
        y = dataset.effect.flatten().numpy()

        log_mg_x, _, _ = compute_marginal_likelihood_nn(x, conf_name = conf_name , flow_name = flow_name, n_steps = n_steps, seed=seed, device=device)
        log_mg_y, _, _ = compute_marginal_likelihood_nn(y, conf_name = conf_name , flow_name = flow_name, n_steps = n_steps, seed=seed, device=device)

        marginal_diff = log_mg_x - log_mg_y
        print(f"successful estimation, sample {idx}")
        return idx, marginal_diff
    except Exception as e:
        print(f"[WARNING] Sample {idx} failed with error: {e}")
        return idx, np.nan


from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
from sklearn.preprocessing import StandardScaler

config_tag = f"s{args.steps}_{args.flow}_{args.conf}"

if __name__ == "__main__":
    for ds_name, ds_class, n_samples in DATASETS:
        print(f"⏳ Estimating dataset: {ds_name} with {n_samples} samples... (steps={args.steps}, workers={MAX_WORKERS})")
        start_time = time.time()
        results = []

        # adjust how many samples to run; here it's all of them
        indices = list(range(1, n_samples + 1))

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    estimate_sample,
                    ds_class = ds_class, idx=idx,
                    n_steps=args.steps, conf_name=args.conf, 
                    flow_name=args.flow, preprocessor=StandardScaler()
                ): idx for idx in indices
            }
            for future in tqdm(as_completed(futures), total=len(indices), desc=ds_name):
                idx, marginal_diff = future.result()
                results.append((idx, marginal_diff))

        ls_all_results_marginals[ds_name] = sorted(results, key=lambda x: x[0])
        df_results = pd.DataFrame(ls_all_results_marginals[ds_name],
                                  columns=["sample_idx","marginal_diff"])
        out_csv = RESULTS_DIR / f"{config_tag}_marginals_{ds_name}.csv"
        df_results.to_csv(out_csv, index=False)
        print(f"✅ Saved results for {ds_name} to '{out_csv}'")

        dur_min = (time.time() - start_time)/60.0
        print(f"⏱️ {ds_name} completed in {dur_min:.2f} minutes.")
        ls_time_results_marginals['name'].append(ds_name)
        ls_time_results_marginals['time'].append(dur_min)

    pd.DataFrame(ls_time_results_marginals).to_csv(
        RESULTS_DIR / f"{config_tag}_marginals_time.csv",
        index=False
    )

