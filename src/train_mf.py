import pandas as pd
import numpy as np
from src.mf import SVD

# ---------- CONFIG ----------
CONFIG = {"n_factors": 40, "lr": 0.004, "reg": 0.02, "epochs": 30}

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


# ---------- TRAIN ON TRAIN ----------
model = SVD(**CONFIG)
model.fit(train)

train_rmse = evaluate(model, train)
val_rmse = evaluate(model, val)

print(f"\nTrain RMSE: {train_rmse:.4f}")
print(f"Val RMSE:   {val_rmse:.4f}")

# ---------- FINAL TRAIN (train + val) ----------
full_data = pd.concat([train, val], ignore_index=True)

final_model = SVD(**CONFIG)
final_model.fit(full_data)

# ---------- SAVE FINAL MODEL ----------
final_model.save("models/mf_model.pkl")
