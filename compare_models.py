from src.evaluate import evaluate_model
from src.recommend import recommend_mf, hybrid_recommend
from src.content import content_based_recommend, build_genre_matrix

import pandas as pd
import pickle


# ---------- LOAD ----------
train_df = pd.read_pickle("data/train.pkl")
test_df = pd.read_pickle("data/test.pkl")

with open("models/mf_model.pkl", "rb") as f:
    model = pickle.load(f)

movie_features, movie_norms = build_genre_matrix(train_df)

popularity_df = pd.read_pickle("results/popularity_top_10.pkl")
top_popular = popularity_df.index.tolist()

user_ids = test_df["user_id"].unique()


# ---------- WRAPPERS ----------

def mf_wrapper(user_id):

    return recommend_mf(model, user_id, train_df, movie_features, top_k=10)


def content_wrapper(user_id):
    recs = content_based_recommend(
        user_id, train_df, movie_features, movie_norms, top_k=10)
    if not recs:
        return top_popular[:10]
    return recs


def hybrid_wrapper(user_id):

    return hybrid_recommend(
        user_id, model, train_df, movie_features, movie_norms, top_popular, alpha=0.9, top_k=10)


# ---------- EVALUATION ----------

print("\nEvaluating MF...")
mf_results = evaluate_model(mf_wrapper, test_df, user_ids, k=10)

print("\nEvaluating Content...")
content_results = evaluate_model(content_wrapper, test_df, user_ids, k=10)

print("\nEvaluating Hybrid...")
hybrid_results = evaluate_model(hybrid_wrapper, test_df, user_ids, k=10)


# ---------- PRINT ----------
def print_results(name, results):
    print(f"\n{name}")
    for k, v in results.items():
        print(f"{k:15}: {v}")


print_results("MF", mf_results)
print_results("Content", content_results)
print_results("Hybrid", hybrid_results)

results_df = pd.DataFrame([
    {"model": "MF", **mf_results},
    {"model": "Content", **content_results},
    {"model": "Hybrid", **hybrid_results},
])

results_df = results_df.round(4)
results_df = results_df[
    ["model", "precision@k", "recall@k", "f1@k", "map@k", "num_users", "skipped_users"]]
results_df.to_csv("results/model_comparison.csv", index=False)

print("\n=== Model Comparison ===")
print(results_df)
