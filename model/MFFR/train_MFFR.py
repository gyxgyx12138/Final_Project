from model.MFFR.stage_1 import *
from model.MFFR.stage_2 import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from math import sqrt

def evaluate_model(U, V, R_test):
    test_indices = np.nonzero(R_test)
    # print(f'Test indices: {test_indices}')
    R_pred = predict_ratings(U, V)
    
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
alpha = 0.01
lambda_ = 0.01
gamma_U = 0.005
gamma_V = 0.005
gamma_P = 0.005
# learning_rate = 0.05
top_n = 5



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
        rmse, mae, f1 = evaluate_model(U, V, R_test)
        # print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss}')
    predicted_ratings = predict_ratings(U, V)
    top_n_recommendations = recommend_top_n(predicted_ratings, top_n)
    # print(top_n_recommendations)
    # Evaluate model
    rmse, mae, f1 = evaluate_model(U, V, R_test)
    return rmse, mae, f1