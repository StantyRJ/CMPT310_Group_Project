import os
import json
import itertools
import time
import random
from datetime import datetime
from multiprocessing import Pool, Manager

from lib.models import SVMClassifier
from lib.pipeline import run_training
from lib.datasets.preloaded import PreloadedDatasetProvider
from lib.datasets.png_dataset import PNGDataset

###############################################################################
# CONFIGURATION
###############################################################################

RESULTS_FILE = "svm_search_results.jsonl"
NUM_WORKERS = 4  # Adjust to your CPU cores

# Reduced search space
KERNEL_VALUES = ['linear', 'polynomial', 'rbf', 'sigmoid']
GAMMA_VALUES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
COEF0_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DEGREE_VALUES = [2, 3]
C_VALUES = [0.1, 1, 10, 100, 1000]
SHRINKING_VALUES = [True]
TOL_VALUES = [1e-2]
CACHE_SIZE_VALUES = [2000]

###############################################################################
# FILE HELPERS
###############################################################################

def load_results():
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results

def store_result(params, accuracy, duration):
    entry = {
        **params,
        "accuracy": accuracy,
        "train_seconds": duration,
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

###############################################################################
# SEARCH SPACE
###############################################################################

def generate_all_valid_combinations():
    combos = []
    for (kernel, C, gamma, coef0, degree, shrinking, tol, cache) in itertools.product(
        KERNEL_VALUES, C_VALUES, GAMMA_VALUES, COEF0_VALUES,
        DEGREE_VALUES, SHRINKING_VALUES, TOL_VALUES, CACHE_SIZE_VALUES
    ):
        if kernel not in ["polynomial", "sigmoid"] and coef0 != 0:
            continue
        if kernel != "polynomial" and degree != 3:
            continue
        combos.append({
            "kernel": kernel,
            "C": C,
            "gamma": gamma,
            "coef0": coef0,
            "degree": degree,
            "shrinking": shrinking,
            "tol": tol,
            "cache_size": cache,
        })
    return combos

###############################################################################
# PRELOAD DATASET
###############################################################################

print("[DATASET] Preloading dataset...")
png_dataset = PNGDataset("data/distorted", test_dir="data/characters", test_fraction=0.6)
X_train, y_train, X_test, y_test = png_dataset.load()
dataset_provider = PreloadedDatasetProvider(X_train, y_train, X_test, y_test)
print("[DATASET] Done preloading.")

###############################################################################
# WORKER FUNCTION
###############################################################################

def train_one(params):
    model = SVMClassifier(**params)
    start = time.time()
    accuracy = run_training(model, dataset_provider)
    duration = time.time() - start
    return params, accuracy, duration

###############################################################################
# SEARCH LOOP USING APPLY_ASYNC
###############################################################################

def search_all_parallel():
    all_combos = generate_all_valid_combinations()
    total = len(all_combos)
    manager = Manager()
    status = manager.list(["idle"] * total)  # Worker statuses

    completed = load_results()
    tested_keys = {tuple(r[k] for k in ["kernel","C","gamma","coef0","degree","shrinking","tol","cache_size"])
                   for r in completed}
    remaining = [c for c in all_combos if tuple(c.values()) not in tested_keys]

    # Randomize the order of remaining combinations so tests run in a random sequence
    random.shuffle(remaining)
    print(f"[SEARCH] Randomized order of {len(remaining)} parameter combinations")

    def callback(result):
        params, accuracy, duration = result
        store_result(params, accuracy, duration)
        idx = remaining.index(params)
        status[idx] = "done"
        print_progress(status, remaining)

    pool = Pool(processes=NUM_WORKERS)
    for params in remaining:
        idx = remaining.index(params)
        status[idx] = "queued"
        pool.apply_async(train_one, args=(params,), callback=callback)

    pool.close()
    pool.join()

def print_progress(status_list, remaining):
    total = len(status_list)
    done = status_list.count("done")
    queued = status_list.count("queued")
    running = status_list.count("running")  # optional: update in callback if desired
    bar_len = 40
    filled = int(done / total * bar_len)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\n[PROGRESS] [{bar}] done={done}/{total}, queued={queued}\n")

###############################################################################
# ENTRY POINT
###############################################################################

if __name__ == "__main__":
    search_all_parallel()
