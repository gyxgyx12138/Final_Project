import csv
import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import ast
import numpy as np
from math import sqrt
import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from helper.general_functions import create_and_write_csv
from model.DeepCGSR.combine_review_rating import Calculate_Deep, Calculate_Deep_Orginal, merge_features, mergeReview_Rating, merge_features_mf
from model.DeepCGSR.init import args
from model.DeepCGSR.data_processing import TransformLabel, merge_csv_columns
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from model.DeepCGSR.review_processing.merge_senmatic_review import extract_features, initialize_features
from helper.utils import read_data, setup_path, word_segment, convert_string_to_float_list
from model.DeepCGSR.review_processing.coarse_gain import get_word2vec_model
from model.DeepCGSR.data_processing import TransformLabel_Deep
from model.DeepCGSR.rating_processing.svd import initialize_svd
from model.DeepCGSR.rating_processing.factorization_machine import run

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path) # best model
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def reprocess_input(data):
    user_idx = torch.tensor([int(x) for x in data['reviewerID']], dtype=torch.long)
    item_idx = torch.tensor([int(x) for x in data['itemID']], dtype=torch.long)
    rating = torch.tensor([float(x) for x in data['overall']], dtype=torch.float32)
    item_bias = torch.tensor([float(x) for x in data['item_bias']], dtype=torch.float32)
    user_bias = torch.tensor([float(x) for x in data['user_bias']], dtype=torch.float32)

    user_feature = []
    for item in data['Udeep']:
        if isinstance(item, str):
            user_feature.append(torch.tensor(ast.literal_eval(item), dtype=torch.float32))
        elif isinstance(item, np.ndarray):
            user_feature.append(torch.tensor(item, dtype=torch.float32))
        else:
            user_feature.append(item.float())
    
    item_feature = []
    for item in data['Ideep']:
        if isinstance(item, str):
            item_feature.append(torch.tensor(ast.literal_eval(item), dtype=torch.float32))
        elif isinstance(item, np.ndarray):
            item_feature.append(torch.tensor(item, dtype=torch.float32))
        else:
            item_feature.append(item.float())
    
    user_feature = torch.stack(user_feature)
    item_feature = torch.stack(item_feature)
    
    return user_idx, item_idx, rating, user_feature, item_feature, item_bias, user_bias


def calculate_rmse(y_true, y_pred):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    squared_errors = (y_true_np - y_pred_np) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse

# Define the model
class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(FullyConnectedModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_indices, item_indices, user_features, item_features, item_bias, user_bias):
        # user_features = torch.tensor(user_features)
        # item_features = torch.tensor(item_features)
        interaction = user_features * item_features
        interaction_sum = interaction.sum(dim=1)
        # print("interaction_sum: ", interaction_sum.size())
        if(len(interaction_sum.size()) != self.input_dim):
            self.fc = nn.Linear(len(interaction_sum), len(interaction_sum), bias=True)
            
        # Multiply by weights
        prediction = self.fc(interaction_sum.to(dtype=torch.float32))
        prediction += self.global_bias + item_bias.squeeze()  + user_bias.squeeze() 
        return prediction.squeeze()

    
def train_deepcgsr(train_data_loader, valid_data_loader, num_factors, batch_size, epochs, method_name, log_interval=100):
    print("=================== Training DeepCGSR model ============================")
    model = FullyConnectedModel(input_dim=batch_size, output_dim=num_factors)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.09)
    early_stopper = EarlyStopper(num_trials=5, save_path=f'model/DeepCGSR/chkpt/{method_name}.pt')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_data_loader):
            try:
                user_idx, item_idx, rating, user_feature, item_feature, item_bias, user_bias = reprocess_input({
                    'reviewerID': batch[0],
                    'itemID': batch[1],
                    'overall': batch[2],
                    'Udeep': batch[3],
                    'Ideep': batch[4],
                    'item_bias': batch[5],
                    'user_bias': batch[6],
                })
                
                predictions = model(user_idx, item_idx, user_feature, item_feature, item_bias, user_bias)
                print("predictions: ", predictions)
                # predictions = torch.clamp(predictions, min=1.0, max=5.0)
                loss = criterion(predictions, rating)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                if (batch_idx + 1) % log_interval == 0:
                    print(f"Train Epoch: {epoch+1} [{batch_idx * len(batch[0])}/{len(train_data_loader.dataset)} "
                          f"({100. * batch_idx / len(train_data_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                    
            except Exception as e:
                print("Error: ", e)

        # In tổng loss sau mỗi epoch
        # print(f"Epoch {epoch+1}: Average Loss: {total_loss/len(train_data_loader)}")

        # Kiểm tra AUC trên tập kiểm tra sau mỗi epoch
        auc = test(model, valid_data_loader) 
        # print(f"Validation AUC: {auc}")
        
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break

    return model



def test(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            # Chuyển đổi batch thành từ điển để sử dụng với reprocess_input
            data = {
                'reviewerID': batch[0],
                'itemID': batch[1],
                'overall': batch[2],
                'Udeep': batch[3],
                'Ideep': batch[4],
                'item_bias': batch[5],
                'user_bias': batch[6],
            }
            
            user_idx, item_idx, target, udeep, ideep, item_bias, user_bias = reprocess_input(data)

            # Đảm bảo udeep và ideep là danh sách các tensor hoặc mảng số thực
            udeep = torch.tensor(udeep, dtype=torch.float32) if isinstance(udeep, list) else udeep
            ideep = torch.tensor(ideep, dtype=torch.float32) if isinstance(ideep, list) else ideep
            # Forward pass
            y = model(user_idx, item_idx, udeep, ideep, item_bias, user_bias)
            
            targets.extend(target)
            predicts.extend([round(float(pred)) for pred in y.flatten().cpu().numpy()])

    new_targets = [-1 if i < 4 else 1 for i in targets]
    new_predicts = [-1 if i < 4 else 1 for i in predicts]

    accuracy = accuracy_score(new_targets, new_predicts)
    # print("Accuracy: ", accuracy)
    return accuracy

def test_rsme(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                        # Chuyển đổi batch thành từ điển để sử dụng với reprocess_input
            data = {
                'reviewerID': batch[0],
                'itemID': batch[1],
                'overall': batch[2],
                'Udeep': batch[3],
                'Ideep': batch[4],
                'item_bias': batch[5],
                'user_bias': batch[6],
            }
            
            user_idx, item_idx, target, udeep, ideep, item_bias, user_bias = reprocess_input(data)
            y = model(user_idx, item_idx, udeep, ideep, item_bias, user_bias)
            targets.extend(target)
            predicts.extend([float(pred) for pred in y.flatten().cpu().numpy()])

    new_targer = []
    new_predict = []
    new_targer = targets
    new_predict = new_predict

    # for i in targets:
    #     if i < 4:
    #         new_targer.append(-1)
    #     else:
    #         new_targer.append(1)
    # for i in predicts:
    #     if i < 4:
    #         new_predict.append(-1)
    #     else:
    #         new_predict.append(1)
    # print("F1_Score: ", f1_score(new_targer, new_predict))
    print("rsme raw: ", calculate_rmse(targets, predicts))
    mae_value = mean_absolute_error(targets, predicts)
    print("MAE: ", mae_value)
    return calculate_rmse(targets, predicts), mae_value

def format_array(arr):
    # Chuyển mảng thành chuỗi với định dạng mong muốn
    return "[" + ", ".join(map(str, arr)) + "]"

def map_and_add_column(df1, df2, column_df1, column_df2=None, column_to_map=None, new_column_name='new_column'):
    # Nếu df2 là DataFrame, chuyển nó thành dict
    if isinstance(df2, pd.DataFrame):
        if column_df2 is None or column_to_map is None:
            raise ValueError("Cần chỉ định column_df2 và column_to_map khi df2 là DataFrame")
        map_dict = df2.set_index(column_df2)[column_to_map].to_dict()
    elif isinstance(df2, dict):
        map_dict = df2
    else:
        raise ValueError("df2 phải là DataFrame hoặc dict")

    # Thêm cột mới vào dataframe đầu tiên với tên tùy chỉnh bằng cách mapping
    df1[new_column_name] = df1[column_df1].map(map_dict)
    
    # Định dạng lại các giá trị trong cột mới nếu chúng là mảng
    df1[new_column_name] = df1[new_column_name].apply(
        lambda x: format_array(x) if isinstance(x, (list, np.ndarray)) else x
    )
    
    return df1

def calulate_user_item_bias(allFeatureReviews):
    allFeatureReviews['Ideep'] = allFeatureReviews['Ideep'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64))
    allFeatureReviews['Udeep'] = allFeatureReviews['Udeep'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64))
    
    # Tạo ma trận các vector đặc trưng từ cột 'Ideep'
    item_features = np.vstack(allFeatureReviews['Ideep'].tolist())
    user_features = np.vstack(allFeatureReviews['Udeep'].tolist())

    # Lấy ratings thực tế từ cột 'overall' và chuyển NaN thành 0
    ratings = np.array(allFeatureReviews['overall'].tolist()).astype(np.float64)
    ratings = np.nan_to_num(ratings, nan=0.0)  # Chuyển NaN thành 0

    # Tính bias cho từng item
    item_bias = calculate_bias(item_features, ratings)
    user_bias = calculate_bias(user_features, ratings)    
    
    return item_bias, user_bias

def calculate_bias(feature_vectors, ratings):    
    # Khởi tạo mô hình hồi quy tuyến tính
    model = LinearRegression()
    model.fit(feature_vectors, ratings)

    predicted_ratings = model.predict(feature_vectors)
    bias = ratings - predicted_ratings
    
    return bias


def calculate_bias_deepcgsr(features, ratings):
    model = LinearRegression()
    model.fit(features, ratings)
    predicted_ratings = model.predict(features)
    bias = ratings - predicted_ratings
    return bias

def parse_array_from_string(array_string):
    try:
        if isinstance(array_string, (int, float)):
            return [float(array_string)]

        array_string = array_string.strip()
        array_string = re.sub(r'(?<![\d.])e[\d.]+', '', array_string)
        return ast.literal_eval(array_string)
    except (ValueError, SyntaxError):
        return []

def csv_to_dataloader(csv_link, batch_size, shuffle=True):
    df = pd.read_csv(csv_link)

    df['Udeep'] = df['Udeep'].apply(parse_array_from_string)
    df['Ideep'] = df['Ideep'].apply(parse_array_from_string)
    
    df['reviewerID'] = df['reviewerID'].astype(int)
    df['itemID'] = df['itemID'].astype(int)
    
    reviewerID_tensor = torch.tensor(df['reviewerID'].values, dtype=torch.long)
    itemID_tensor = torch.tensor(df['itemID'].values, dtype=torch.long)
    overall_tensor = torch.tensor(df['overall'].values, dtype=torch.float32)
    Udeep_tensor = torch.tensor(df['Udeep'].tolist(), dtype=torch.float32)
    Ideep_tensor = torch.tensor(df['Ideep'].tolist(), dtype=torch.float32)
    itembias_tensor = torch.tensor(df['item_bias'].values, dtype=torch.float32)
    userbias_tensor = torch.tensor(df['user_bias'].values, dtype=torch.float32)
    
    dataset = TensorDataset(reviewerID_tensor, itemID_tensor, overall_tensor, Udeep_tensor, Ideep_tensor, itembias_tensor, userbias_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle)
    
    return dataloader


def encode_and_save_csv(df, output_path, columns_to_encode):
    # Khởi tạo LabelEncoder
    label_encoders = {}
    
    # Mã hóa từng cột được chỉ định
    for column in columns_to_encode:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    
    # Lưu DataFrame dưới dạng file CSV
    df.to_csv(output_path, index=False)
    
    return label_encoders


window_size = 3
min_count = 1
vector_size = 200
# model_path = "model/DeepCGSR/output/word2vec.model"
is_train = True # 是否训练


# DeepCGSR
def DeepCGSR(dataset_df, num_factors, num_words, filename, method_name="DeepCGSR", is_switch_data=False):

    if method_name == "DeepCGSR":
        train_data_list = dataset_df["reviewText"].tolist()
    else:
        train_data_list = dataset_df["filteredReviewText"].tolist()
        
    model_path = "model/DeepCGSR/output/word2vec" + method_name + ".model"   
    allreviews_path, reviewer_path, item_path, _, _, _, _, final_data_path, svd_path, checkpoint_path, sparse_matrix_path = setup_path(method_name)   
    split_data = []
    # print("train_data_list: ", train_data_list)
    for i in train_data_list:
        split_data.append(word_segment(i))
    word2vec_model = get_word2vec_model(is_train=is_train,
                            model_path=model_path,
                            split_data=split_data,
                            vector_size=vector_size,
                            min_count=min_count,
                            window=window_size)

    allFeatureReviews = extract_features(dataset_df, split_data, word2vec_model, num_factors, num_words, filename, method_name, is_switch_data)
    reviewer_feature_dict, item_feature_dict = initialize_features(filename, num_factors, method_name)
    
    svd = initialize_svd(allreviews_path + filename + ".csv", num_factors, svd_path + filename +'.pt')
    z_item, review_item_list, rating_item_list = mergeReview_Rating(item_path + filename +".csv", "z_item_" + filename, svd, reviewer_feature_dict, item_feature_dict, "item", method_name)
    z_review, review_user_list, rating_user_list = mergeReview_Rating(reviewer_path + filename +".csv", "z_reviewer_" + filename, svd, reviewer_feature_dict, item_feature_dict, "reviewer", method_name)
    
    v_reviewer_list = []
    v_item_list = []
    fm = run(allreviews_path + filename +".csv", num_factors * 2, checkpoint_path + filename +'.pkl', sparse_matrix_path + filename +'.npz')
    for name in z_review.items():
        v_reviewer_list.append(fm.get_embedding('reviewerID_' + name[0]))

    for name in z_item.items():
        v_item_list.append(fm.get_embedding('itemID_' + name[0]))
        
        
    print("================")
    u_deep = {}
    i_deep = {}
    if method_name == "DeepCGSR":
        for (z_name, z_value), v_value  in zip(z_review.items(), v_reviewer_list):
            u_deep[z_name] = Calculate_Deep_Orginal(z_value, v_value)

        for (z_name, z_value), v_value in zip(z_item.items(), v_item_list):
            i_deep[z_name] = Calculate_Deep_Orginal(z_value, v_value)
    else:
        # u_deep = Calculate_Deep(v_list, z_review, 0)
        # i_deep = Calculate_Deep(v_list, z_item, len(z_review))
        for (z_name, z_value), v_value  in zip(z_review.items(), v_reviewer_list):
            u_deep[z_name] = merge_features_mf(z_value, v_value, num_factors * 2)

        for (z_name, z_value), v_value in zip(z_item.items(), v_item_list):
            i_deep[z_name] = merge_features_mf(z_value, v_value, num_factors * 2)
        
        
        # svd = initialize_svd(allreviews_path + filename + ".csv", num_factors * 2, svd_path + filename +'.pt')
        # for (z_name, z_value), review_value, rating_value  in zip(z_review.items(), review_user_list, rating_user_list):
        #     u_deep[z_name] = merge_features_mf(z_value, svd.get_user_embedding(z_name), 32)

        # for (z_name, z_value), review_value, rating_value in zip(z_item.items(), review_item_list, rating_item_list):
        #     i_deep[z_name] = merge_features_mf(z_value, svd.get_item_embedding(z_name), 32)
        
    create_and_write_csv("u_deep_" + filename, u_deep, method_name)
    create_and_write_csv("i_deep_" + filename, i_deep, method_name)

    allFeatureReviews = allFeatureReviews[['reviewerID', 'itemID', 'overall']]
    allFeatureReviews = map_and_add_column(allFeatureReviews, u_deep, 'reviewerID', 'Key', 'Array', 'Udeep')
    allFeatureReviews = map_and_add_column(allFeatureReviews, i_deep, 'itemID', 'Key', 'Array', 'Ideep')
    
    if method_name == "DeepCGSR":
        # Tính user_bias dựa trên Udeep
        user_biases = {}
        for reviewer_id, group in allFeatureReviews.groupby('reviewerID'):
            user_features = group['Udeep'].values.reshape(-1, 1)
            ratings = group['overall'].values
            user_bias = calculate_bias(user_features, ratings)
            user_biases[reviewer_id] = user_bias.mean()

        # Tính item_bias dựa trên Ideep
        item_biases = {}
        for item_id, group in allFeatureReviews.groupby('itemID'):
            item_features = group['Ideep'].values.reshape(-1, 1)
            ratings = group['overall'].values
            item_bias = calculate_bias(item_features, ratings)
            item_biases[item_id] = item_bias.mean()
            
        allFeatureReviews['user_bias'] = allFeatureReviews['reviewerID'].map(user_biases)
        allFeatureReviews['item_bias'] = allFeatureReviews['itemID'].map(item_biases)
    else:
        item_bias, user_bias = calulate_user_item_bias(allFeatureReviews)
        # Thêm bias vào DataFrame
        allFeatureReviews['item_bias'] = item_bias
        allFeatureReviews['user_bias'] = user_bias
                # Tính user_bias dựa trên Udeep
        # user_biases = {}
        # for reviewer_id, group in allFeatureReviews.groupby('reviewerID'):
        #     user_features = group['Udeep'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64)).values
        #     print("user_features: ", user_features)
        #     ratings = group['overall'].values
        #     # ratings = np.full(user_features, ratings_avg)  # Tạo mảng với cùng số lượng mẫu
        #     user_bias = calculate_bias(user_features, ratings)
        #     user_biases[reviewer_id] = user_bias
            
        # # Tính item_bias dựa trên Ideep
        # item_biases = {}
        # for item_id, group in allFeatureReviews.groupby('itemID'):
        #     item_features = group['Ideep'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64)).values
        #     ratings = group['overall'].values
        #     # ratings = np.full(item_features, ratings_avg)  # Tạo mảng với cùng số lượng mẫu
        #     item_bias = calculate_bias(item_features, ratings)
        #     item_biases[item_id] = item_bias

            
        # allFeatureReviews['user_bias'] = allFeatureReviews['reviewerID'].map(user_biases)
        # allFeatureReviews['item_bias'] = allFeatureReviews['itemID'].map(item_biases)

    # print(allFeatureReviews[['item_bias']])
    # print(allFeatureReviews[['user_bias']])
    
    encode_and_save_csv(allFeatureReviews, final_data_path + method_name + "_" + filename +".csv", ['reviewerID', 'itemID'])
    


