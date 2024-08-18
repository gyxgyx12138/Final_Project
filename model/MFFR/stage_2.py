import numpy as np

# Step 1: Initialize Matrices
def initialize_matrices(n_users, n_items, n_factors):
    U = np.random.normal(scale=1./n_factors, size=(n_users, n_factors))
    V = np.random.normal(scale=1./n_factors, size=(n_items, n_factors))
    P = np.random.normal(scale=1./n_factors, size=(n_items, n_factors))
    return U, V, P

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

# Step 4: Calculate the Predicted Rating Matrix
def predict_ratings(U, V):
    return np.dot(U, V.T)

# Step 5: Generate Top-N Recommendations
def recommend_top_n(predicted_ratings, N):
    top_n_recommendations = np.argsort(-predicted_ratings, axis=1)[:, :N]
    return top_n_recommendations


