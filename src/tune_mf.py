import pandas as pd
import numpy as np
from itertools import product
from src.mf import SVD


# ---------- LOAD ----------
train = pd.read_pickle("data/train.pkl")
val = pd.read_pickle("data/val.pkl")


# ---------- EVALUATION ----------
def evaluate(model, df):
    errors = []

    for row in df.itertuples():
        pred = model.predict(row.user_id, row.movie_id)
        err = row.rating - pred
        errors.append(err ** 2)

    return np.sqrt(np.mean(errors))


# ---------- SEARCH SPACE ----------
param_grid = {
    "n_factors": [30, 40, 50],
    "lr": [0.003, 0.004, 0.005],
    "reg": [0.01, 0.02],
    "epochs": [30]
}

keys = list(param_grid.keys())
values = list(param_grid.values())
experiments = list(product(*values))


# ---------- RUN EXPERIMENTS ----------
results = []

best_val_rmse = float("inf")
best_config = None

for exp in experiments:
    config = dict(zip(keys, exp))
    print(f"\nRunning: {config}")

    model = SVD(**config)
    model.fit(train)

    train_rmse = evaluate(model, train)
    val_rmse = evaluate(model, val)

    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Val RMSE:   {val_rmse:.4f}")

    row = {**config, "train_rmse": train_rmse, "val_rmse": val_rmse}
    results.append(row)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_config = config


# ---------- SAVE RESULTS ----------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("val_rmse")  # sort best → worst

results_df.to_csv("results/mf_tuning_results.csv", index=False)


# ---------- PRINT SUMMARY ----------
print("\n===== TOP RESULTS =====")
print(results_df.head())

print("\nBest Config:", best_config)
print(f"Best Val RMSE: {best_val_rmse:.4f}")
