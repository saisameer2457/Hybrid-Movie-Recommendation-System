import streamlit as st
import pandas as pd
import numpy as np
import pickle

from src.recommend import hybrid_recommend
from src.content import build_genre_matrix


# ---------- CLEAN TITLES ----------
def clean_title(title):
    # Fix ", The", ", A", ", An"
    if ", The" in title:
        title = "The " + title.replace(", The", "")
    elif ", A" in title:
        title = "A " + title.replace(", A", "")
    elif ", An" in title:
        title = "An " + title.replace(", An", "")

    # Remove leading apostrophes
    title = title.lstrip("'")

    # Clean leading dots (...)
    if title.startswith("..."):
        title = title[3:].strip()

    return title

# ---------- LOAD ----------


@st.cache_data
def load_data():
    train_df = pd.read_pickle("data/train.pkl")
    popularity_df = pd.read_pickle("results/popularity_top_10.pkl")

    movie_features, movie_norms = build_genre_matrix(train_df)

    raw_titles = (
        train_df.drop_duplicates("movie_id").set_index(
            "movie_id")["title"].to_dict()
    )

    movie_titles = {mid: clean_title(t) for mid, t in raw_titles.items()}

    title_to_id = {v: k for k, v in movie_titles.items()}

    movie_genres = (
        train_df.drop_duplicates("movie_id")
        .set_index("movie_id")["genres"]
        .to_dict()
    )

    top_popular = popularity_df.index.tolist()
    user_ids = sorted(train_df["user_id"].unique())

    return (train_df, movie_features, movie_norms, movie_titles,
            title_to_id, movie_genres, top_popular, user_ids
            )


@st.cache_resource
def load_model():
    with open("models/mf_model.pkl", "rb") as f:
        return pickle.load(f)


(train_df, movie_features, movie_norms, movie_titles,
 title_to_id, movie_genres, top_popular, user_ids,
 ) = load_data()

model = load_model()

# ---------- UI ----------
st.set_page_config(page_title="Hybrid Movie Recommender", layout="centered")

st.title("🎬 Hybrid Movie Recommendation System")
st.caption("Matrix Factorization + Content-Based Filtering")
mode = st.radio(
    "Choose Recommendation Mode",
    ["👤 Personalized (Hybrid)", "🔍 Explore Movies (Content-Based)"]
)

# ---------- PERSONALIZED ----------
if mode == "👤 Personalized (Hybrid)":
    user_id = st.selectbox("Select User", user_ids)

    def get_recommendations():
        return hybrid_recommend(
            user_id, model, train_df, movie_features,
            movie_norms, top_popular, alpha=0.9, top_k=10
        )


# ---------- ITEM BASED ----------
else:
    movie_list = sorted(title_to_id.keys())
    selected_movie = st.selectbox("🎬 Select a movie", movie_list)

    def get_recommendations():
        movie_id = title_to_id.get(selected_movie)

        if movie_id is None:
            return top_popular[:10]

        # --- CONTENT-BASED ITEM SIMILARITY ---

        target_vec = movie_features.get(movie_id)
        target_norm = movie_norms.get(movie_id)

        if target_vec is None or target_norm == 0:
            return top_popular[:10]

        scores = []

        for m_id, vec in movie_features.items():
            if m_id == movie_id:
                continue

            denom = target_norm * movie_norms[m_id]
            if denom == 0:
                continue

            sim = np.dot(target_vec, vec) / denom
            scores.append((m_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in scores[:10]]


# ---------- BUTTON ----------
if st.button("🎯 Recommend"):
    with st.spinner("Generating recommendations..."):
        recs = get_recommendations()

    st.subheader("Top Recommendations")

    for i, movie_id in enumerate(recs, 1):
        title = movie_titles.get(movie_id, f"Movie {movie_id}")
        genres = movie_genres.get(movie_id, "")

        st.markdown(f"**{i}. 🎬 {title}**")
        if genres:
            st.markdown(f"🎭 {', '.join(genres)}")

    st.success("Done")
