import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from helper.general_functions import save_to_excel
from helper.utils import dataloader_to_dataframe, read_and_process_csv, read_data, word_segment, backup_and_delete_files
# from model.DeepCGSR.train import DeepCGSR
from model.DeepCGSR.review_processing.coarse_gain import get_word2vec_model
from model.DeepCGSR.train import DeepCGSR, csv_to_dataloader, test, test_rsme, train_deepcgsr
from model.MFFR.train_MFFR import evaluate_MFFR, MFFR
from model.MFFR.SAMF import update_ratings_with_sentiment, get_sentiment_scores, load_bert_model


# from model.DeepCGSR.train import train

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

# Hàm để tạo DataLoader cho các tập train, validation, test
def create_dataloaders(json_file, batch_size=32, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    dataset = read_data(json_file)
    total_size = len(dataset)
    
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = total_size - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

# Chia dữ liệu thành các DataFrame (train, validation, test)
def create_dataframes(json_file, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2):
    # Đọc dữ liệu từ tệp JSON
    data = read_data(json_file)
    df = pd.DataFrame(data, columns=['reviewerID', 'asin', 'overall', 'overall_new', 'reviewText', 'filteredReviewText'])
    
    # Tách dữ liệu thành tập train và temp (validation + test)
    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=42)
    
    # Tính toán tỷ lệ của tập validation so với tập temp
    valid_ratio_temp = valid_ratio / (valid_ratio + test_ratio)
    
    # Tách tập temp thành tập validation và test
    valid_df, test_df = train_test_split(temp_df, train_size=valid_ratio_temp, random_state=42)
    
    return df, train_df, valid_df, test_df

# Ví dụ sử dụng
dataset_name = "Small_All_Beauty_5_0"
json_file = "model/DeepCGSR/data/" + dataset_name + ".json"
batch_size = 32
num_epochs = 100
# num_factors = 16
list_factors = [40, 8]
num_words = 300
is_switch_data = True
rsme_MFFR = 0
mae_MFFR = 0
f1_MFFR = 0
for num_factors in list_factors:
    #for i in range(1):    
    
    # train_loader, valid_loader, test_loader = create_dataloaders(json_file, batch_size)
    all_df, train_df, valid_df, test_df = create_dataframes(json_file)

    #region DeepCGSR 

    method_name = ["triet_method", "DeepCGSR"]
    for method in method_name:
        print("Method: ", method)
        
        DeepCGSR(train_df, num_factors, num_words, "train", method, is_switch_data)
        DeepCGSR(valid_df, num_factors, num_words, "vaild", method, is_switch_data)
        DeepCGSR(test_df, num_factors, num_words, "test", method, is_switch_data)

        final_feature_train_path = "model/DeepCGSR/data/final_data_feature_" + method + "_train.csv"
        final_feature_valid_path = "model/DeepCGSR/data/final_data_feature_" + method + "_train.csv"
        final_feature_test_path = "model/DeepCGSR/data/final_data_feature_" + method + "_test.csv"

        train_data_loader = csv_to_dataloader(final_feature_train_path, batch_size)
        valid_data_loader = csv_to_dataloader(final_feature_valid_path, batch_size)
        test_data_loader = csv_to_dataloader(final_feature_test_path, batch_size)
        
        model_deep = train_deepcgsr(train_data_loader, valid_data_loader, num_factors, batch_size, num_epochs, method, log_interval=100)
        auc_test = test(model_deep, test_data_loader)
        rsme_test, mae_test = test_rsme(model_deep, test_data_loader)
        DeepCGSR_results = [auc_test, rsme_test, mae_test]
        
        save_to_excel([DeepCGSR_results], ['AUC', 'RSME Test', 'MAE Test'], "model/results/"+ method + "_" + dataset_name + "_factors" + str(num_factors) + ".xlsx")
    #endregion

    # #region MFFR
    # method_name = ["MFFR", "SAMF"]
    # for method in method_name:
    #     rsme_MFFR = 0
    #     mae_MFFR = 0
    #     f1_MFFR = 0
    #     loop = 4
    #     for i in range(loop):
    #         print("Method: ", method)
    #         test_df = pd.concat([test_df, valid_df], ignore_index=True)
    #         # train_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)
    #         predicted_ratings, R_test = MFFR(train_df, test_df, num_factors, 100)
    #         if method == "SAMF":
    #             print("Training Sentiment Model")
    #             tokenizer, sentiment_model = load_bert_model()
    #             sentiment_scores = get_sentiment_scores(train_df['reviewText'].astype(str), tokenizer, sentiment_model)
    #             predicted_ratings = update_ratings_with_sentiment(predicted_ratings, sentiment_scores)
    #             print("Updated Rating Matrix:", predicted_ratings)
    #         rsme_new, mae_new, f1_new = evaluate_MFFR(predicted_ratings, R_test)
    #         rsme_MFFR += rsme_new
    #         mae_MFFR += mae_new
    #         f1_MFFR += f1_new
    #         # MFFR_results = [f1_new, rsme_new, mae_new]
    #     MFFR_results = [f1_MFFR/loop, rsme_MFFR/loop, mae_MFFR/loop]    
    #     save_to_excel([MFFR_results], ['AUC', 'RSME Test', 'MAE'], "model/results/"+  method + "_" + dataset_name + "_factors" + str(num_factors) + ".xlsx")
    # #endregion
        
    backup_and_delete_files("model/DeepCGSR/feature", "model/DeepCGSR/backup", "BKfeature", "290824", extensions=[".csv"])
    backup_and_delete_files("model/DeepCGSR/feature_originalmethod", "model/DeepCGSR/backup", "BKfeature_originalmethod", "290824", extensions=[".csv"])
    backup_and_delete_files("model/DeepCGSR/data", "model/DeepCGSR/backup", "BKdata", "290824", extensions=[".csv"])
    backup_and_delete_files("model/DeepCGSR/chkpt", "model/DeepCGSR/backup", "BK_chkpt", "290824", True, extensions=[".pt", ".pkl", "npz"])
    backup_and_delete_files("model/DeepCGSR/output", "model/DeepCGSR/backup", "BK_output", "290824", extensions=[".model"])


