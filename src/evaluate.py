import numpy as np


def precision_at_k(recommended, relevant, k):
    if not recommended:
        return 0.0

    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / min(k, len(recommended))


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / min(len(relevant), k)


def f1_at_k(recommended, relevant, k):
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)

    if (p + r) == 0:
        return 0.0

    return 2 * (p * r) / (p + r)


def average_precision_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    relevant_set = set(relevant)
    hits = 0
    score = 0.0

    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(relevant), k)


def map_at_k(recommended_list, relevant_list, k):
    return np.mean([
        average_precision_at_k(rec, rel, k)
        for rec, rel in zip(recommended_list, relevant_list)
    ]) if recommended_list else 0.0


def evaluate_model(recommend_fn, test_df, user_ids, k=10):
    precisions, recalls, f1s = [], [], []
    recommended_all, relevant_all = [], []

    skipped_users = 0

    test_group = test_df.groupby("user_id")

    for user_id in user_ids:
        if user_id not in test_group.groups:
            skipped_users += 1
            continue

        relevant = test_group.get_group(user_id)
        relevant = relevant[relevant["rating"] >= 4]["movie_id"].tolist()

        if not relevant:
            skipped_users += 1
            continue

        recommended = recommend_fn(user_id)

        if not recommended:
            skipped_users += 1
            continue

        p = precision_at_k(recommended, relevant, k)
        r = recall_at_k(recommended, relevant, k)
        f1 = f1_at_k(recommended, relevant, k)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        recommended_all.append(recommended)
        relevant_all.append(relevant)

    return {
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "f1@k": float(np.mean(f1s)) if f1s else 0.0,
        "map@k": float(map_at_k(recommended_all, relevant_all, k)),
        "num_users": int(len(precisions)),
        "skipped_users": int(skipped_users)
    }
