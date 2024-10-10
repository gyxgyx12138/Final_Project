import json
import string
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import torch
import json
import ast
import re
import os
import glob
import shutil

def read_and_process_csv(file_path):
    # Đọc file CSV
    df = pd.read_csv(file_path)
    
    # Kiểm tra các cột có tồn tại trong DataFrame không
    required_columns = ['UserId', 'ProductId', 'Score', 'Text']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Kiểm tra giá trị của cột Score, chỉ giữ lại những giá trị từ 1 đến 5
    df = df[df['Score'].between(1, 5)]
    
    # Chuyển đổi cột review_text từ dạng list sang string nếu cần thiết
    def list_to_string(review_text):
        if isinstance(review_text, str) and review_text.startswith('[') and review_text.endswith(']'):
            try:
                # Nếu review_text là một biểu diễn của list, chuyển đổi nó
                review_list = ast.literal_eval(review_text)
                # Nếu review_list thực sự là list, chuyển thành string
                if isinstance(review_list, list):
                    return ' '.join(review_list)
            except (ValueError, SyntaxError):
                # Nếu không phải là list, trả lại chuỗi ban đầu
                return review_text
        return review_text  # Nếu không phải là list hoặc không cần chuyển đổi, trả lại chuỗi ban đầu
    
    df['Text'] = df['Text'].apply(list_to_string)
    
    # Đổi tên các cột
    df = df.rename(columns={
        'UserId': 'reviewerID',
        'ProductId': 'asin',
        'Score': 'overall',
        'Text': 'reviewText'
    })
    
    return df

def read_data(file_path):
    """
       params:
           file_path: Đường dẫn đến tệp
       return:
           data: Danh sách dữ liệu đã đọc, mỗi dòng là một mẫu

    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                # Chuyển đổi từ văn bản JSON sang đối tượng Python
                raw_sample = json.loads(line)
                if 'reviewText' not in raw_sample:
                    raw_sample['reviewText'] = ''
                # Chuẩn hóa dữ liệu và thêm vào danh sách data
                data.append([raw_sample['reviewerID'],
                             raw_sample['asin'],
                             raw_sample['overall'],
                             raw_sample['overall_new'],
                             raw_sample['reviewText'],
                             raw_sample['filteredReviewText']])
            except json.JSONDecodeError:
                # Bắt các lỗi khi chuyển đổi từ JSON
                # Bạn có thể xử lý các lỗi ở đây nếu cần
                pass
    return data



def softmax(x):
    """Compute the softmax of vector x.
    """
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def sigmoid(x):
    x = np.clip(x, -709, 709)
    s = 1 / (1 + np.exp(-x))
    return s


# 加载停用词
stop_words = stopwords.words("english") + list(string.punctuation)
def word_segment(text):
    # Kiểm tra nếu text là None hoặc NaN
    if pd.isnull(text):
        return []  # Trả về list rỗng nếu text là None hoặc NaN
    # Tokenize chuỗi text
    word_seg = word_tokenize(str(text).lower())  # Chuyển về chữ thường và tokenize
    
    return word_seg


def preprocessed(text):
    """ 3文本预处理
    """
    # 分句和词性还原， 目前只实现分句
    return text.split("\.")

def clean_text(text):
    # Loại bỏ các dấu chấm liên tiếp
    text = re.sub(r'\.{2,}', ' ', text)
    return text

def tensor_to_list(data):
    if isinstance(data, torch.Tensor):
        return data.numpy().tolist()
    elif isinstance(data, (list, tuple)):
        return [tensor_to_list(item) for item in data]
    else:
        return data

# Hàm chuyển đổi DataLoader về DataFrame
def dataloader_to_dataframe(dataloader):
    data_list = []

    for batch in dataloader:
        batch_data = tensor_to_list(batch)
        data_list.extend(batch_data)

    # Tạo DataFrame từ danh sách
    if isinstance(data_list[0], (list, tuple)) and len(data_list[0]) == 2:
        df = pd.DataFrame(data_list, columns=['input', 'target'])
    else:
        df = pd.DataFrame(data_list)
    
    return df

def convert_string_to_float_list(string):
    try:
        # Sử dụng ast.literal_eval để chuyển đổi chuỗi thành danh sách thực
        return np.array(ast.literal_eval(string), dtype=np.float64)
    except:
        # Trả về một mảng rỗng nếu không thể chuyển đổi
        return np.array([])
    
def backup_and_delete_files(folder_path, backup_path, backup_folder_name, date, removeModelBERT = False, extensions=[".csv"]):
    # Tạo đường dẫn tới thư mục sao lưu
    backup_folder_path = os.path.join(backup_path, backup_folder_name + "_" + date)
    print(f"Thư mục sao lưu: {backup_folder_path}")
    
    # Tạo thư mục sao lưu nếu chưa tồn tại
    if not os.path.exists(backup_folder_path):
        print(f"Tạo thư mục sao lưu mới: {backup_folder_path}")
        os.makedirs(backup_folder_path)
    
    # Lặp qua từng loại file extension để sao lưu và xóa các file tương ứng
    for ext in extensions:
        # Tạo danh sách các tệp với phần mở rộng cụ thể trong thư mục gốc
        files = glob.glob(os.path.join(folder_path, f"*{ext}"))

        # Sao chép các tệp vào thư mục sao lưu và sau đó xóa chúng khỏi thư mục gốc
        for file_path in files:
            try:
                file_name = os.path.basename(file_path)
                # Sao chép tệp vào thư mục sao lưu
                shutil.copy(file_path, backup_folder_path)
                print(f"Đã sao chép: {file_path} tới {backup_folder_path}")
                if file_name == "bert_last_checkpoint.pt" and removeModelBERT == False:
                    print(f"Bỏ qua không sao lưu và xóa: {file_path}")
                    continue
                # Xóa tệp từ thư mục gốc
                os.remove(file_path)
                print(f"Đã xóa: {file_path}")
            except Exception as e:
                print(f"Không thể sao chép hoặc xóa {file_path}. Lỗi: {e}")

def setup_path(method_name):
    allreviews_path = "model/DeepCGSR/feature/allFeatureReview_"
    reviewer_path = "model/DeepCGSR/feature/reviewer_feature_"
    item_path = "model/DeepCGSR/feature/item_feature_"
    udeep_path = "model/DeepCGSR/feature/u_deep_"
    ideep_path = "model/DeepCGSR/feature/i_deep_"
    tranformed_udeep_path = "model/DeepCGSR/feature/transformed_udeep_"
    tranformed_ideep_path = "model/DeepCGSR/feature/transformed_ideep_"
    
    final_data_path = "model/DeepCGSR/data/final_data_feature_"
    svd_path = "model/DeepCGSR/chkpt/svd_"
    checkpoint_path = 'model/DeepCGSR/chkpt/fm_checkpoint_'
    sparse_matrix_path = 'model/DeepCGSR/chkpt/encoded_features_'
    
    if method_name == "DeepCGSR":
        allreviews_path = "model/DeepCGSR/feature_originalmethod/allFeatureReview_"
        reviewer_path = "model/DeepCGSR/feature_originalmethod/reviewer_feature_"
        item_path = "model/DeepCGSR/feature_originalmethod/item_feature_"
        udeep_path = "model/DeepCGSR/feature_originalmethod/u_deep_"
        ideep_path = "model/DeepCGSR/feature_originalmethod/i_deep_"
        tranformed_udeep_path = "model/DeepCGSR/feature_originalmethod/transformed_udeep_"
        tranformed_ideep_path = "model/DeepCGSR/feature_originalmethod/transformed_ideep_"
        checkpoint_path =  checkpoint_path + "feature_originalmethod"
        sparse_matrix_path = sparse_matrix_path + "feature_originalmethod"
        svd_path = svd_path + "feature_originalmethod"
        final_data_path = final_data_path
    return allreviews_path, reviewer_path, item_path, udeep_path, ideep_path, tranformed_udeep_path, tranformed_ideep_path, final_data_path, svd_path, checkpoint_path, sparse_matrix_path

if __name__ == "__main__":

    import nltk
    nltk.download('averaged_perceptron_tagger')
    # from nltk.corpus import sentiwordnet as swn
    # breakdown = swn.senti_synset('breakdown.n.03')
    # print(breakdown)