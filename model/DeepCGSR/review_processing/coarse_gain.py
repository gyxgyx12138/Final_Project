from model.DeepCGSR.review_processing.fine_gain import get_word_sentiment_score_addition, get_word_sentiment_score
from helper.utils import softmax, word_segment, sigmoid
from gensim.models import word2vec, Word2Vec
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch
import os


def get_word2vec_model(is_train, model_path, split_data=None, vector_size=None, min_count=None, window=None):
    """word2vec训练代码
    """
    if is_train:
        model = word2vec.Word2Vec(split_data, vector_size=vector_size, min_count=min_count, window=window)
        model.save(model_path)
    else:
        model = Word2Vec.load(model_path)
    return model

def get_coarse_simtiment_score(text, word2vec_model):
    word_seg = word_segment(text)
    sim_word = []
    sim_word_weight = []
    for e in word2vec_model.wv.most_similar(positive=word_seg, topn=10):
        sim_word.append(e[0])
        sim_word_weight.append(e[1])
    return sim_word, softmax(sim_word_weight)


def get_coarse_sentiment_score(model, tokenizer, text):
    # Thiết lập thiết bị (GPU nếu có, nếu không dùng CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Đảm bảo rằng mô hình được chuyển sang thiết bị
    model = model.to(device)

    # Tokenize văn bản và chuyển tensor đầu vào sang thiết bị
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    # Dự đoán kết quả
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Logits: các giá trị thô từ mô hình
    logits = outputs.logits
    
    # Tính xác suất cho các lớp
    probabilities = torch.softmax(logits, dim=1)
    
    # Tìm lớp có xác suất cao nhất
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Thang điểm cảm xúc tương ứng với các lớp
    sentiment_scale = [(0.0, 0.19), (0.2, 0.39), (0.4, 0.59), (0.6, 0.79), (0.8, 0.99)]
    
    # Xác suất cao nhất
    highest_prob = probabilities[0][predicted_class].item()
    
    # Ánh xạ xác suất về khoảng [lower_bound, upper_bound]
    lower_bound, upper_bound = sentiment_scale[predicted_class]
    sentiment_score = lower_bound + highest_prob * (upper_bound - lower_bound)
    
    return sentiment_score


# Get coarse-grained sentiment score
def get_coarse_score(text, word2vec_model):
    word_seg = word_segment(text)
    sim_word, sim_word_weight = get_coarse_simtiment_score(text, word2vec_model)
    score = 0
    for i, j in zip(sim_word, sim_word_weight):
        score += get_word_sentiment_score_addition(i) * j
    return sigmoid(score)

def get_coarse_score_LDA(text, word2vec_model):
    word_seg = word_segment(text)
    sim_word, sim_word_weight = get_coarse_simtiment_score(text, word2vec_model)
    score = 0
    for i, j in zip(sim_word, sim_word_weight):
        score += get_word_sentiment_score(i) * j
    return sigmoid(score)

