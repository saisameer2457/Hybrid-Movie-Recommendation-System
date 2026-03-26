import numpy as np


def build_genre_matrix(df):
    all_genres = set()
    for genres in df["genres"]:
        all_genres.update(genres)

    genre_list = sorted(all_genres)
    genre_index = {g: i for i, g in enumerate(genre_list)}

    movie_features = {}

    for row in df.itertuples():
        if row.movie_id in movie_features:
            continue  # avoid rebuilding same movie

        vec = np.zeros(len(genre_list), dtype=np.float32)

        for g in row.genres:
            vec[genre_index[g]] = 1.0

        movie_features[row.movie_id] = vec

    # Precompute norms (optimization)
    movie_norms = {
        m: np.linalg.norm(v) for m, v in movie_features.items()
    }

    return movie_features, movie_norms


def build_user_profile(user_id, train_df, movie_features):
    user_data = train_df[train_df["user_id"] == user_id]

    profile = None
    total_weight = 0.0

    for row in user_data.itertuples():
        vec = movie_features.get(row.movie_id)
        if vec is None:
            continue

        # ⭐ Better weighting (centered rating)
        weight = row.rating - 2.5
        if weight <= 0:
            continue

        if profile is None:
            profile = np.zeros_like(vec)

        profile += vec * weight
        total_weight += weight

    if profile is None or total_weight == 0:
        return None

    return profile / total_weight


def content_based_recommend(user_id, train_df, movie_features, movie_norms, top_k=10):
    user_profile = build_user_profile(user_id, train_df, movie_features)

    if user_profile is None:
        return []

    user_norm = np.linalg.norm(user_profile)
    if user_norm == 0:
        return []

    watched = set(
        train_df[train_df["user_id"] == user_id]["movie_id"]
    )

    scores = []

    # ⚡ Fast cosine similarity using precomputed norms
    for movie_id, vec in movie_features.items():
        if movie_id in watched:
            continue

        denom = user_norm * movie_norms[movie_id]
        if denom == 0:
            continue

        sim = np.dot(user_profile, vec) / denom
        scores.append((movie_id, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    return [movie_id for movie_id, _ in scores[:top_k]]
