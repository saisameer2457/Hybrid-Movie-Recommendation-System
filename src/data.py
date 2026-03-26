import pandas as pd


def load_data():
    ratings = pd.read_csv(
        "data/ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        "data/movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1"
    )

    return ratings, movies


def merge_data(ratings, movies):
    return ratings.merge(movies, on="movie_id")


def preprocess_data(df, min_user_ratings=5, min_movie_ratings=5):
    df = df.copy()

    # Filter users
    user_counts = df["user_id"].value_counts()
    df = df[df["user_id"].isin(
        user_counts[user_counts >= min_user_ratings].index)]

    # Filter movies
    movie_counts = df["movie_id"].value_counts()
    df = df[df["movie_id"].isin(
        movie_counts[movie_counts >= min_movie_ratings].index)]

    # Convert genres to list
    df["genres"] = df["genres"].apply(lambda x: x.split("|"))

    return df


def per_user_time_split(df, train_ratio=0.7, val_ratio=0.1):
    train_list, val_list, test_list = [], [], []

    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("timestamp")
        n = len(user_df)

        if n < 5:
            train_list.append(user_df)
            continue

        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        train_list.append(user_df.iloc[:train_end])
        val_list.append(user_df.iloc[train_end:val_end])
        test_list.append(user_df.iloc[val_end:])

    train = pd.concat(train_list).reset_index(drop=True)
    val = pd.concat(val_list).reset_index(drop=True)
    test = pd.concat(test_list).reset_index(drop=True)

    # Drop timestamp
    train = train.drop(columns=["timestamp"])
    val = val.drop(columns=["timestamp"], errors="ignore")
    test = test.drop(columns=["timestamp"], errors="ignore")

    # Preserve column order
    cols = ["user_id", "movie_id", "rating", "title", "genres"]
    train = train[cols]
    val = val[cols]
    test = test[cols]
    return train, val, test


def save_data(train, val, test):
    train.to_pickle("data/train.pkl")
    val.to_pickle("data/val.pkl")
    test.to_pickle("data/test.pkl")


if __name__ == "__main__":
    ratings, movies = load_data()
    df = merge_data(ratings, movies)
    df = preprocess_data(df)

    train, val, test = per_user_time_split(df)

    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

    save_data(train, val, test)
