import numpy as np

# Step 1: Initialize Matrices
def initialize_matrices(n_users, n_items, n_factors):
    U = np.random.normal(scale=1./n_factors, size=(n_users, n_factors))
    V = np.random.normal(scale=1./n_factors, size=(n_items, n_factors))
    P = np.random.normal(scale=1./n_factors, size=(n_items, n_factors))
    return U, V, P
# def initialize_matrices(n_users, n_items, n_factors, n_preferences):
#     # Initialize U, V, P with small random values
#     U = np.random.randn(n_users, n_factors) * 0.01
#     V = np.random.randn(n_items, n_factors) * 0.01
#     P = np.random.randn(n_preferences, n_factors) * 0.01
#     return U, V, P


def compute_loss(R, S, U, V, P, alpha, lambda_):
    prediction_loss = np.sum((R[R > 0] - np.dot(U, V.T)[R > 0]) ** 2)
    preference_loss = alpha * np.sum((S[S > 0] - np.dot(U, P.T)[S > 0]) ** 2)
    regularization = lambda_ * (np.sum(U**2) + np.sum(V**2) + np.sum(P**2))
    return prediction_loss + preference_loss + regularization

def sgd_update(R, S, U, V, P, gamma_U, gamma_V, gamma_P, alpha, lambda_):
    n_users, n_items = R.shape
    for i in range(n_users):
        for j in range(n_items):
            if R[i, j] > 0:
                eij = R[i, j] - np.dot(U[i, :], V[j, :])
                U[i, :] += gamma_U * (eij * V[j, :] - lambda_ * U[i, :])
                V[j, :] += gamma_V * (eij * U[i, :] - lambda_ * V[j, :])
                
            if S[i, j] > 0:
                sij = S[i, j]
                eij_s = sij - np.dot(U[i, :], P[j, :])
                U[i, :] += gamma_U * alpha * (eij_s * P[j, :] - lambda_ * U[i, :])
                P[j, :] += gamma_P * alpha * (eij_s * U[i, :] - lambda_ * P[j, :])
    
    loss = compute_loss(R, S, U, V, P, alpha, lambda_)

    return U, V, P, loss


class SGD:
    def __init__(self, lr=0.01, epochs=1000, batch_size=32, tol=1e-3):
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tol
        self.weights = None
        self.bias = None

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, X_batch, y_batch):
        y_pred = self.predict(X_batch)
        error = y_pred - y_batch
        gradient_weights = np.dot(X_batch.T, error) / X_batch.shape[0]
        gradient_bias = np.mean(error)
        return gradient_weights, gradient_bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                gradient_weights, gradient_bias = self.gradient(X_batch, y_batch)
                self.weights -= self.learning_rate * gradient_weights
                self.bias -= self.learning_rate * gradient_bias

            if epoch % 100 == 0:
                y_pred = self.predict(X)
                loss = self.mean_squared_error(y, y_pred)
                print(f"Epoch {epoch}: Loss {loss}")

            if np.linalg.norm(gradient_weights) < self.tolerance:
                print("Convergence reached.")
                break

        return self.weights, self.bias
    

class SGD_MatrixFactorization:
    def __init__(self, n_factors, lr=0.01, epochs=1000, batch_size=32, tol=1e-3, alpha=0.1, lambda_1=0.1, lambda_2=0.1, lambda_3=0.1):
        self.n_factors = n_factors
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tol
        self.alpha = alpha  # Coefficient for topic feature loss
        self.lambda_1 = lambda_1  # Regularization for U
        self.lambda_2 = lambda_2  # Regularization for V
        self.lambda_3 = lambda_3  # Regularization for P
        self.U = None
        self.V = None
        self.P = None

    def predict_rating(self, U, V):
        return np.dot(U, V.T)

    def predict_feature(self, U, P):
        return np.dot(U, P.T)

    def g(self, x):
        return 1 / (1 + np.exp(-x))

    def g_prime(self, x):
        return self.g(x) * (1 - self.g(x))  # Derivative of sigmoid
    
    def gradient_U(self, R, S, U, V, P):
        dL_dU = np.zeros_like(U)
        n_users, n_items = R.shape
        n_topics = S.shape[1]

        for i in range(n_users):
            # Gradient for the rating term
            for j in range(n_items):
                if R[i, j] > 0:  # Only for observed ratings
                    pred_rating = self.g(np.dot(U[i].T, V[j]))  # Use activation function g
                    dL_dU[i] += -(self.g(R[i, j]) - pred_rating) * V[j]
                    
            # Gradient for the topic feature term
            for k in range(n_topics):
                if S[i, k] > 0:  # Only for observed topics
                    pred_feature = self.g(np.dot(U[i].T, P[k]))  # Use activation function g
                    dL_dU[i] += -self.alpha * (self.g(S[i, k]) - pred_feature) *  P[k]  # Apply chain rule
                    #dL_dU[i] += -self.alpha * (S[i, k]) - pred_feature * P[k]
                    
            # Regularization term for U
            dL_dU[i] += self.lambda_1 * U[i]
        dL_dU = dL_dU + U * self.lambda_1
        return dL_dU

    def gradient_V(self, R, U, V):
        dL_dV = np.zeros_like(V)
        n_users, n_items = R.shape
        
        for j in range(n_items):
            for i in range(n_users):
                if R[i, j] > 0:  # Only for observed ratings
                    if(R[i, j] > 3):
                        R[i, j] = 1
                    else:
                        R[i, j] = 0
                    pred_rating = self.g(np.dot(U[i].T, V[j]))  # Use activation function g
                    dL_dV[j] += -(R[i, j] - pred_rating) * U[i]  # Apply chain rule
                    #dL_dV[j] += -(R[i, j] - pred_rating) * U[i]

            # Regularization term for V
            dL_dV[j] += self.lambda_2 * V[j]
        dL_dV = dL_dV + V * self.lambda_2
        return dL_dV

    def gradient_P(self, S, U, P):
        dL_dP = np.zeros_like(P)
        n_users, n_topics = S.shape

        for k in range(n_topics):
            for i in range(n_users):
                if S[i, k] > 0:  # Only for observed topics
                    pred_feature = self.g(np.dot(U[i].T, P[k]))  # Use activation function g
                    dL_dP[k] += -self.alpha * (self.g(S[i, k]) - pred_feature) *  U[i]  # Apply chain rule
                    #dL_dP[k] += -self.alpha * ((S[i, k]) - pred_feature) * U[i]
            # Regularization term for P
            dL_dP[k] += self.lambda_3 * P[k]
        dL_dP = dL_dP + P * self.lambda_3
        return dL_dP
    
    # def gradient_U(self, R, S, U, V, P):
    #     dL_dU = np.zeros_like(U)
    #     n_users, n_items = R.shape
    #     n_topics = S.shape[1]

    #     for i in range(n_users):
    #         # Gradient for the rating term
    #         for j in range(n_items):
    #             if R[i, j] > 0:  # Only for observed ratings
    #                 pred_rating = np.dot(U[i], V[j])  # Raw dot product, no sigmoid
    #                 dL_dU[i] += -(R[i, j] - pred_rating) * V[j]  # No g(R[i, j])

    #         # Gradient for the topic feature term
    #         for k in range(n_topics):
    #             if S[i, k] > 0:  # Only for observed topics
    #                 pred_feature = np.dot(U[i], P[k])  # Raw dot product, no sigmoid
    #                 dL_dU[i] += -self.alpha * (S[i, k] - pred_feature) * P[k]  # No g(S[i, k])

    #         # Regularization term for U
    #         dL_dU[i] += self.lambda_1 * U[i]

    #     return dL_dU

    # def gradient_V(self, R, U, V):
    #     dL_dV = np.zeros_like(V)
    #     n_users, n_items = R.shape
        
    #     for j in range(n_items):
    #         for i in range(n_users):
    #             if R[i, j] > 0:  # Only for observed ratings
    #                 pred_rating = np.dot(U[i], V[j])  # Raw dot product, no sigmoid
    #                 dL_dV[j] += -(R[i, j] - pred_rating) * U[i]  # No g(R[i, j])

    #         # Regularization term for V
    #         dL_dV[j] += self.lambda_2 * V[j]

    #     return dL_dV

    # def gradient_P(self, S, U, P):
    #     dL_dP = np.zeros_like(P)
    #     n_users, n_topics = S.shape

    #     for k in range(n_topics):
    #         for i in range(n_users):
    #             if S[i, k] > 0:  # Only for observed topics
    #                 pred_feature = np.dot(U[i], P[k])  # Raw dot product, no sigmoid
    #                 dL_dP[k] += -self.alpha * (S[i, k] - pred_feature) * U[i]  # No g(S[i, k])

    #         # Regularization term for P
    #         dL_dP[k] += self.lambda_3 * P[k]

    #     return dL_dP


    def fit(self, R, S):
        n_users, n_items = R.shape
        n_topics = S.shape[1]

        # Initialize U, V, and P matrices
        self.U = np.random.uniform(1, 5, (n_users, self.n_factors))
        self.V = np.random.uniform(1, 5, (n_items, self.n_factors))
        self.P = np.random.uniform(1, 5, (n_topics, self.n_factors))


        for epoch in range(self.epochs):
            # Shuffle users
            indices = np.random.permutation(n_users)
            R_shuffled = R[indices]
            S_shuffled = S[indices]

            for i in range(0, n_users, self.batch_size):
                U_batch = self.U[indices[i:i+self.batch_size]]
                R_batch = R_shuffled[i:i+self.batch_size]
                S_batch = S_shuffled[i:i+self.batch_size]

                # Compute gradients
                dL_dU = self.gradient_U(R_batch, S_batch, U_batch, self.V, self.P)
                dL_dV = self.gradient_V(R_batch, U_batch, self.V)
                dL_dP = self.gradient_P(S_batch, U_batch, self.P)

                # Update U, V, P
                self.U[indices[i:i+self.batch_size]] -= self.learning_rate * dL_dU
                self.V -= self.learning_rate * dL_dV
                self.P -= self.learning_rate * dL_dP

            if epoch % 100 == 0:
                pred_ratings = self.predict_rating(self.U, self.V)
                print("pred_ratings: ", pred_ratings)
                loss = np.mean((R - pred_ratings) ** 2)
                print(f"Epoch {epoch}: Loss {loss}")

            # Check convergence
            if np.linalg.norm(dL_dU) < self.tolerance:
                print("Convergence reached.")
                break

        return self.U, self.V, self.P

# Step 4: Calculate the Predicted Rating Matrix
def predict_ratings(U, V):
    return np.dot(U, V.T)

# Step 5: Generate Top-N Recommendations
def recommend_top_n(predicted_ratings, N):
    top_n_recommendations = np.argsort(-predicted_ratings, axis=1)[:, :N]
    return top_n_recommendations


