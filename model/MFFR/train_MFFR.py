from model.MFFR.stage_1 import *
from model.MFFR.stage_2 import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from math import sqrt

# Sigmoid function as a non-linear mapping 'g'
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
    
def convert_to_range_1_5(x):
    # Convert a value from the range [0, 1] to the range [1, 5]
    return 4 * x + 1

def evaluate_model(R_pred, R_test):
    test_indices = np.nonzero(R_test)
    # print(f'Test indices: {test_indices}')

    print(f'R_pred: {R_pred}')
    # Lấy các giá trị tại các chỉ số có rating trong R_test
    R_test_nonzero = R_test[test_indices]
    R_pred_nonzero = R_pred[test_indices]

    # Lọc ra các giá trị mà người dùng đánh giá cao (overall > 3)
    R_test_nonzero_filtered = np.where(R_test_nonzero > 3, 1, 0)
    R_pred_nonzero_filtered = np.where(R_pred_nonzero > 0, 1, 0)
    
    if len(R_test_nonzero_filtered) == 0:
        raise ValueError("Test set has no entries with rating greater than 3.")
    
    rmse = sqrt(mean_squared_error(R_test_nonzero_filtered, R_pred_nonzero_filtered))
    mae = mean_absolute_error(R_test_nonzero_filtered, R_pred_nonzero_filtered)
    f1 = f1_score(R_test_nonzero_filtered, R_pred_nonzero_filtered)
    print(f'RMSE: {rmse}, MAE: {mae}, F1: {f1}')
    return rmse, mae, f1

def compute_loss(U, V, P, R_train, S, lambda_U, lambda_V, lambda_P, alpha_u):
    # Rating loss (MSE) for ratings R
    rating_loss = 0
    for i in range(U.shape[0]):
        for j in range(V.shape[0]):
            if R_train[i, j] != 0:
                predicted_rating = sigmoid(np.dot(U[i], V[j]))
                rating_loss += (R_train[i, j] - predicted_rating) ** 2

    # Preference loss (MSE) for preferences S
    preference_loss = 0
    for i in range(U.shape[0]):
        for l in range(P.shape[0]):
            if S[i, l] != 0:
                predicted_preference = sigmoid(np.dot(U[i], P[l]))
                preference_loss += (S[i, l] - predicted_preference) ** 2

    # Regularization terms
    reg_loss = (lambda_U * np.sum(U ** 2)) + (lambda_V * np.sum(V ** 2)) + (lambda_P * np.sum(P ** 2))

    # Total loss (including weighted preference loss)
    total_loss = rating_loss + alpha_u * preference_loss + reg_loss
    return total_loss


def precision_at_k(U, V, R_test, k):
    user_count = R_test.shape[0]
    correct_predictions = 0
    total_predictions = user_count * k
    R_pred = predict_ratings(U, V)
    for user in range(user_count):
        actual_items = np.where(R_test[user] > 0)[0]
        predicted_items = np.argsort(R_pred[user])[-k:][::-1]
        correct_predictions += len(set(actual_items) & set(predicted_items))
    precision = correct_predictions / total_predictions
    return precision

def convert_and_save_dataset(input_df, output_file_path):
    # Bước 2: Kiểm tra xem các cột có tồn tại trong dataframe hay không
    if 'reviewerID' not in input_df.columns or 'asin' not in input_df.columns or 'overall' not in input_df.columns:
        raise ValueError("Dataset must contain 'reviewerID', 'asin', and 'rating' columns")
    
    input_df['reviewerID'] = input_df['reviewerID'].astype(str)
    input_df['asin'] = input_df['asin'].astype(str)
    # Bước 3: Mã hóa các cột reviewerID và asin
    reviewer_encoder = LabelEncoder()
    asin_encoder = LabelEncoder()
    
    input_df['reviewerID'] = reviewer_encoder.fit_transform(input_df['reviewerID'])
    input_df['asin'] = asin_encoder.fit_transform(input_df['asin'])
    
    # Bước 4: Lưu DataFrame đã được mã hóa thành tệp CSV mới
    input_df.to_csv(output_file_path, index=False)
    
    # Tạo ma trận rating
    n_users = input_df['reviewerID'].nunique()
    n_items = input_df['asin'].nunique()
    
    rating_matrix = np.zeros((n_users, n_items))
    
    for row in input_df.itertuples():
        rating_matrix[row.reviewerID, row.asin] = row.overall
    
    return rating_matrix, input_df, n_users, n_items


# Example usage
n_users = 0
n_items = 0


# Hyperparameters
batch_size = 32
alpha_u = 0.5
lambda_ = 0.001
gamma_U = 0.01
gamma_V = 0.01
gamma_P = 0.01
lambda_U = 0.1
lambda_V = 0.1
lambda_P = 0.1
alpha = 0.03

def get_test_predictions(predicted_ratings, df_test, user_index_mapping, item_index_mapping, R_test_shape):
    # Extract user-item pairs from df_test
    user_ids = df_test['reviewerID'].map(user_index_mapping)
    item_ids = df_test['asin'].map(item_index_mapping)
    
    # Check for any unmapped IDs (NaNs) and filter them out
    mask = (~user_ids.isna()) & (~item_ids.isna())
    user_ids = user_ids[mask].astype(int).to_numpy()
    item_ids = item_ids[mask].astype(int).to_numpy()

    # Initialize an empty array to store predictions with the same shape as R_test
    test_predictions = np.zeros(R_test_shape)

    # Ensure `predicted_ratings` is 2-dimensional
    if predicted_ratings.ndim != 2:
        raise ValueError("predicted_ratings must be a 2-dimensional array (matrix)")

    # Fill `test_predictions` using the indices from `user_ids` and `item_ids`
    for i in range(len(user_ids)):
        test_predictions[user_ids[i], item_ids[i]] = predicted_ratings[user_ids[i], item_ids[i]]

    return test_predictions



def MFFR(train_df, test_df, n_factors, n_epochs):
    item_review_list = train_df.groupby("asin")["reviewText"].apply(list)
    user_review_list = train_df.groupby("reviewerID")["reviewText"].apply(list)
    item_topic_features = extract_topic_features(item_review_list, n_factors)
    user_topic_features = extract_topic_features(user_review_list, n_factors)
    
    print("User topic features: ", user_topic_features.shape)
    print("Item topic features: ", item_topic_features.shape)
    
    S = construct_preference_matrix(user_topic_features, item_topic_features) # preference matrix
    
    R_test, df, n_users, n_items = convert_and_save_dataset(test_df, "model/MFFR/data/test.csv") # rating matrix
    R_train, df, n_users, n_items = convert_and_save_dataset(train_df, "model/MFFR/data/train.csv") # rating matrix
    

    U, V, P = initialize_matrices(n_users, n_items, n_factors)
    for epoch in range(n_epochs):
        U_new, V_new, P_new, loss = sgd_update(R_train, S, U, V, P, gamma_U, gamma_V, gamma_P, alpha, lambda_)
        U = U_new
        V = V_new
        P = P_new
        # rmse, mae, f1 = evaluate_model(U, V, R_test)
        # print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss}')
    predicted_ratings = predict_ratings(U, V)

    return predicted_ratings, R_test

def evaluate_MFFR(predicted_ratings, R_test):
    # top_n_recommendations = recommend_top_n(predicted_ratings, top_n)
    # print(top_n_recommendations)
    # Evaluate model
    rmse, mae, f1 = evaluate_model(predicted_ratings, R_test)
    return rmse, mae, f1 