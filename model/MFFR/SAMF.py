import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', num_labels=5)
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)


# BERT tokenization and embedding extraction using CLS token
# def get_bert_embeddings(reviews):
#     # Đảm bảo mọi phần tử trong reviews là chuỗi
#     if isinstance(reviews, pd.Series):  # Nếu là pandas Series, chuyển thành list
#         reviews = reviews.astype(str).tolist()
#     elif isinstance(reviews, list):
#         reviews = [str(review) for review in reviews]  # Chuyển từng phần tử thành chuỗi
#     elif isinstance(reviews, str):
#         reviews = [reviews]  # Nếu là chuỗi đơn, chuyển thành danh sách 1 phần tử
#     print("Reviews: ", reviews)
#     inputs = tokenizer(reviews, return_tensors='tf', padding=True, truncation=True)
#     outputs = bert_model(**inputs)
#     cls_embedding = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token's embedding
#     return cls_embedding
    # Tokenize and get BERT embeddings for each document
def get_bert_embeddings(texts, batch_size=32):
    texts = [str(text) for text in texts]
    texts = texts.to_list() if isinstance(texts, pd.Series) else texts
    
    # Split the input texts into batches
    bert_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        inputs = tokenizer(batch_texts, return_tensors='tf', padding=True, truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = bert_model(**inputs)  # Use BERT to get embeddings
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embedding
            
        bert_embeddings.append(cls_embedding)
    
    # Concatenate all batch embeddings into one tensor
    return tf.concat(bert_embeddings, axis=0)




# Sentiment Analysis using BiRNN
class SentimentRNN(tf.keras.Model):
    def __init__(self):
        super(SentimentRNN, self).__init__()
        self.bi_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        # Reshape inputs to (batch_size, sequence_length, feature_size)
        x = tf.expand_dims(inputs, axis=1)  # Add sequence length dimension of 1
        x = self.bi_rnn(x)
        return self.dense(x)

# Create a simple dataset (replace this with actual sentiment labels for your data)
# For example, positive (1) and negative (0) labels for sentiment analysis
# labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # Example labels corresponding to the reviews

def train_sentiment_model(reviews, labels):
    bert_embeddings = get_bert_embeddings(reviews)

    # Convert BERT embeddings from tensor to numpy array for splitting
    bert_embeddings = bert_embeddings.numpy()
    assert bert_embeddings.shape[0] == len(labels), "Embedding count does not match label count"

    
    # Initialize and compile the sentiment model
    sentiment_model = SentimentRNN()
    sentiment_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model (replace with more epochs and proper dataset for better results)
    sentiment_model.fit(bert_embeddings, labels, epochs=10, batch_size=2)
    
    return sentiment_model


# # Now use the trained model to predict sentiment scores on new reviews
# new_reviews = ["I love this product!", "good product", "Quality is amazing, but delivery was late."]

def get_sentiment_scores(test_reviews, sentiment_model):
    # Get BERT embeddings for new reviews
    new_bert_embeddings = get_bert_embeddings(test_reviews)
    new_bert_embeddings = new_bert_embeddings.numpy()

    # Predict sentiment scores using the trained model
    sentiment_scores = sentiment_model.predict(new_bert_embeddings)
    print(f'Predicted sentiment scores: {sentiment_scores}')
    return sentiment_scores




# Sentiment integration as described in the paper
alpha = 0.5  # Weight for sentiment adjustment

# Adjust ratings with sentiment scores
def update_ratings_with_sentiment(rating_matrix, sentiment_scores, alpha=0.1):
    updated_ratings = np.copy(rating_matrix)
    for i in range(len(rating_matrix)):
        for j in range(len(rating_matrix[i])):
            if rating_matrix[i, j] != 0:
                updated_ratings[i, j] = (1 - alpha) * rating_matrix[i, j] + alpha * sentiment_scores[i]
    return updated_ratings

# Apply sentiment-based adjustment



