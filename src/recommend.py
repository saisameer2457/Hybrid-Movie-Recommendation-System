import numpy as np
from src.content import content_based_recommend, build_user_profile


def recommend_mf(model, user_id, train_df, movie_features, top_k=10):
    user_data = train_df[train_df["user_id"] == user_id]
    watched = set(user_data["movie_id"])

    scores = []

    # Use movie_features keys for consistency
    for movie_id in movie_features:
        if movie_id in watched:
            continue

        score = model.predict(user_id, movie_id)
        scores.append((movie_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    return [movie_id for movie_id, _ in scores[:top_k]]


def hybrid_recommend(
        user_id, model, train_df, movie_features, movie_norms, top_popular, alpha=0.9, top_k=10):
    user_data = train_df[train_df["user_id"] == user_id]
    watched = set(user_data["movie_id"])

    user_profile = build_user_profile(user_id, train_df, movie_features)

    if user_profile is None:
        return top_popular[:top_k]

    user_norm = np.linalg.norm(user_profile)
    if user_norm == 0:
        return top_popular[:top_k]

    scores = []
    mf_dict = []
    mf_values = []

    # ⚡ Single pass: compute MF scores once
    for movie_id in movie_features:
        if movie_id in watched:
            continue

        mf = model.predict(user_id, movie_id)
        mf_dict.append((movie_id, mf))
        mf_values.append(mf)

    if not mf_values:
        return top_popular[:top_k]

    # ⚡ Compute hybrid score
    for movie_id, mf in mf_dict:
        vec = movie_features[movie_id]

        mf_norm = (mf - 1) / 4

        # Content score
        denom = user_norm * movie_norms[movie_id]
        if denom == 0:
            continue

        content_score = np.dot(user_profile, vec) / denom

        final_score = alpha * mf_norm + (1 - alpha) * content_score
        scores.append((movie_id, final_score))

    scores.sort(key=lambda x: x[1], reverse=True)

    if not scores:
        return top_popular[:top_k]

    return [movie_id for movie_id, _ in scores[:top_k]]


def recommend(user_id, model, train_df, top_popular, movie_features, movie_norms, top_k=10):
    user_data = train_df[train_df["user_id"] == user_id]

    #  Cold start
    if len(user_data) == 0:
        return top_popular[:top_k]

    # Few interactions → Content-based
    elif len(user_data) < 5:
        cb = content_based_recommend(
            user_id, train_df, movie_features, movie_norms, top_k
        )

        if not cb:
            return top_popular[:top_k]

        return cb

    # Normal → Hybrid
    return hybrid_recommend(
        user_id, model, train_df, movie_features, movie_norms, top_popular, alpha=0.9, top_k=top_k
    )
