import pandas as pd


def compute_popularity(df, top_k=10, save_path="results"):
    stats = df.groupby("movie_id").agg({
        "rating": ["mean", "count"]
    })

    stats.columns = ["avg_rating", "count"]

    C = stats["avg_rating"].mean()
    m = stats["count"].quantile(0.9)

    stats["score"] = (
        (stats["count"] / (stats["count"] + m)) * stats["avg_rating"]
        + (m / (stats["count"] + m)) * C
    )

    stats = stats.sort_values("score", ascending=False)

    stats = stats.head(top_k)
    filename = f"{save_path}/popularity_top_{top_k}.pkl"
    stats.to_pickle(filename)

    return stats


train_df = pd.read_pickle("data/train.pkl")

popularity_df = compute_popularity(train_df, top_k=10, save_path="results")

print("Saved popularity rankings")
print("\nTop movies:")
print(popularity_df)
