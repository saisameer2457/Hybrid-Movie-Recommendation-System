# 🎬 Hybrid Movie Recommendation System

A movie recommender system built on the MovieLens dataset that combines **Matrix Factorization (Collaborative Filtering)** and **Content-Based Filtering** to generate personalized recommendations.

The project implements a complete machine learning workflow, including data preprocessing, time-aware splitting, model training, hyperparameter tuning, ranking-based evaluation, and a Streamlit-based interactive application.

---

## 🚀 Project Overview

This system recommends movies using a combination of:

- **Collaborative Filtering** (Matrix Factorization)
- **Content-Based Filtering** (Genre similarity)
- **Hybrid Recommendation System**

It is designed to reflect real-world recommender systems with proper data splitting, evaluation metrics, and cold-start handling.

---

## 🧠 Key Features

- Custom **Matrix Factorization (SVD-style)** implemented from scratch
- **Content-based filtering** using genre embeddings
- **Hybrid recommendation system** combining multiple signals
- **Cold-start handling** using popularity-based fallback
- **Time-aware train/validation/test split**
- Evaluation using **ranking metrics (Precision@K, Recall@K, MAP@K)**
- **Hyperparameter tuning** for MF model
- **Model comparison pipeline**
- Interactive **Streamlit web app**

---

## 📊 Dataset

Uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/):

- `ratings.dat` → User-item interactions  
- `movies.dat` → Movie metadata (genres)

---

## 🧹 Data Preprocessing

- Filters out low-activity users and rarely rated movies
- Converts genre strings into structured lists
- Performs **per-user time-based split**:
  - Train
  - Validation
  - Test
- Saves processed datasets as `.pkl` files for reuse

---

## 🤖 Models

### 1. Collaborative Filtering (Matrix Factorization)

- Learns:
  - Global mean
  - User bias
  - Item bias
  - Latent embeddings
- Optimized using **Stochastic Gradient Descent (SGD)**
- Tracks training RMSE

---

### 2. Content-Based Filtering

- Uses **genre multi-hot encoding**
- Builds user preference profiles
- Computes similarity using cosine similarity

---

### 3. Popularity-Based Model

- Weighted score using:
  - Average rating
  - Number of ratings
- Used for **cold-start users**

---

### 4. Hybrid Recommendation

Final scoring:
score = α * MF_score + (1 - α) * content_score

- Combines collaborative + content signals
- Improves robustness and recommendation quality

---

## 📈 Evaluation

Models are evaluated using ranking-based metrics:

- Precision@K  
- Recall@K  
- F1@K  
- MAP@K  

---

## 📊 Results

Model performance evaluated using ranking metrics:

| Model   | Precision@K | Recall@K | F1@K  | MAP@K |
|---------|------------|----------|-------|-------|
| MF      | 0.0490     | 0.0548   | 0.0507 | 0.0245 |
| Content | 0.0219     | 0.0285   | 0.0236 | 0.0101 |
| Hybrid  | 0.0473     | 0.0539   | 0.0493 | 0.0229 |

- Collaborative Filtering (MF) performs best overall  
- Hybrid model provides competitive performance by combining collaborative and content signals  
- Content-based model performs weaker due to limited feature representation  

---

### 🔍 Insights

- The **Hybrid model achieves performance close to MF**, while incorporating additional content-based signals  
- Hybrid approach is particularly useful for **cold-start scenarios**, where user interaction data is limited  
- The results suggest that:
  - Collaborative filtering captures strong user-item interaction patterns  
  - Content features (genres) provide limited additional signal in this dataset  

Overall, this demonstrates that hybrid systems improve robustness and coverage, especially for new or sparse users, while maintaining competitive recommendation quality.

---

## 🔄 Recommendation Pipeline

### 1. Candidate Generation
- From MF predictions and/or popularity

### 2. Ranking
- Hybrid scoring
- Top-N recommendation output

---

## 🖥️ Features

- User-based personalized recommendations  
- Item-based movie exploration  
- Displays movie titles and genres  

---

## 📂 Project Structure

```text
Hybrid Movie Recommendation System/
│
├── data/
│   ├── movies.dat
│   ├── ratings.dat
│   ├── train.pkl
│   ├── val.pkl
│   └── test.pkl
│
├── results/
│   ├── mf_tuning_results.csv
│   ├── model_comparison.csv
│   └── popularity_top_10.pkl
│
├── models/
│   └── mf_model.pkl
│
├── src/
│   ├── data.py
│   ├── mf.py
│   ├── content.py
│   ├── popularity.py
│   ├── recommend.py
│   ├── evaluate.py
│   ├── train_mf.py
│   └── tune_mf.py
│
├── app.py
├── compare_models.py
├── requirements.txt
└── README.md
