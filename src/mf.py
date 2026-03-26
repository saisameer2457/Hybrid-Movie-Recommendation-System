import numpy as np
import pickle


class SVD:
    def __init__(self, n_factors=30, lr=0.005, reg=0.02, epochs=20, seed=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.seed = seed

    def fit(self, df):
        np.random.seed(self.seed)

        # Unique users and items
        user_ids = df["user_id"].unique()
        item_ids = df["movie_id"].unique()

        # Create mappings
        self.user_map = {u: i for i, u in enumerate(user_ids)}
        self.item_map = {i: j for j, i in enumerate(item_ids)}

        self.n_users = len(user_ids)
        self.n_items = len(item_ids)

        # Global mean
        self.global_mean = df["rating"].mean()

        # Initialize biases
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)

        # Initialize latent factors
        self.P = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (self.n_items, self.n_factors))

        # 🔥 Precompute arrays (FAST)
        user_indices = df["user_id"].map(self.user_map).values
        item_indices = df["movie_id"].map(self.item_map).values
        ratings = df["rating"].values

        n_samples = len(ratings)

        # Training loop
        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            total_loss = 0

            for idx in indices:
                u = user_indices[idx]
                i = item_indices[idx]
                r = ratings[idx]

                # Prediction
                pred = (
                    self.global_mean
                    + self.user_bias[u]
                    + self.item_bias[i]
                    + np.dot(self.P[u], self.Q[i])
                )

                err = r - pred

                # Update biases
                self.user_bias[u] += self.lr * \
                    (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * \
                    (err - self.reg * self.item_bias[i])

                # Save copy for simultaneous update
                P_u = self.P[u].copy()

                # Update latent factors
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * P_u - self.reg * self.Q[i])

                total_loss += err ** 2

            rmse = np.sqrt(total_loss / n_samples)
            print(f"Epoch {epoch+1}/{self.epochs}, RMSE: {rmse:.4f}")

    def predict(self, user_id, movie_id):
        if user_id not in self.user_map or movie_id not in self.item_map:
            return self.global_mean

        u = self.user_map[user_id]
        i = self.item_map[movie_id]

        pred = (
            self.global_mean
            + self.user_bias[u]
            + self.item_bias[i]
            + np.dot(self.P[u], self.Q[i])
        )

        return np.clip(pred, 1, 5)

    def save(self, path="models/mf_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path="models/mf_model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
