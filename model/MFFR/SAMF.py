import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load pre-trained BERT tokenizer and model (corrected without num_labels in tokenizer)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Function to load the BERT model and tokenizer (corrected)
def load_bert_model(model_name='bert-base-uncased', num_labels=2):
    # Load pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Load pre-trained BERT model for sequence classification
    bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    return tokenizer, bert_model

# Function to get BERT embeddings
def get_bert_embeddings(reviews, tokenizer, bert_model):
    # Ensure all elements in reviews are strings
    if isinstance(reviews, pd.Series):  # If pandas Series, convert to list
        reviews = reviews.astype(str).tolist()
    elif isinstance(reviews, list):
        reviews = [str(review) for review in reviews]  # Convert each element to string
    elif isinstance(reviews, str):
        reviews = [reviews]  # If a single string, convert to a list with one element
    
    # Tokenize and obtain BERT embeddings
    inputs = tokenizer(reviews, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.logits  # Take logits as representation
    return cls_embedding

# Sentiment Analysis model using BiRNN
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

def get_sentiment_scores(test_reviews, tokenizer, sentiment_model, batch_size=8, device='cpu'):
    """
    Generate sentiment scores for new reviews using BERT embeddings and a trained sentiment model.
    
    Args:
        test_reviews: A list or series of new review texts.
        tokenizer: The tokenizer associated with the BERT model.
        sentiment_model: A pre-trained sentiment model that can predict sentiment scores.
        batch_size: Number of reviews to process at a time (to prevent memory overflow).
        device: Device to run the model ('cpu' or 'cuda').
        
    Returns:
        sentiment_scores: Predicted sentiment scores (probabilities) from the sentiment model.
    """
    # Move model to the correct device
    sentiment_model.to(device)
    sentiment_model.eval()  # Set model to evaluation mode
    
    # Ensure test_reviews is a list of strings
    if isinstance(test_reviews, pd.Series):
        test_reviews = test_reviews.tolist()
    elif isinstance(test_reviews, str):
        test_reviews = [test_reviews]
    
    sentiment_scores = []

    # Process reviews in batches to manage memory
    for i in range(0, len(test_reviews), batch_size):
        batch_reviews = test_reviews[i:i+batch_size]
        
        # Tokenize the input reviews
        inputs = tokenizer(batch_reviews, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Move tensors to the correct device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Get model predictions (logits)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)

        # Get the logits (unnormalized predictions)
        logits = outputs.logits

        # Apply softmax to get sentiment probabilities
        batch_scores = torch.nn.functional.softmax(logits, dim=-1).cpu()  # Move results to CPU if needed
        
        # Collect scores
        sentiment_scores.extend(batch_scores.tolist())
    
    return sentiment_scores


# Sentiment-based rating adjustment
alpha = 0.5  # Weight for sentiment adjustment

def update_ratings_with_sentiment(rating_matrix, sentiment_scores, alpha=0.1):
    updated_ratings = np.copy(rating_matrix)
    for i in range(len(rating_matrix)):
        for j in range(len(rating_matrix[i])):
            if rating_matrix[i, j] != 0:
                # Ensure sentiment_scores[i] is a scalar or reduce it to one
                score = sentiment_scores[i]
                
                # If `score` is a list or sequence, reduce it to its mean value
                if isinstance(score, (list, np.ndarray, torch.Tensor)):
                    sentiment_score_scalar = np.mean(score) if not torch.is_tensor(score) else score.mean().item()
                else:
                    sentiment_score_scalar = score
                
                # Safeguard: Ensure `sentiment_score_scalar` is a float or int
                if not isinstance(sentiment_score_scalar, (float, int)):
                    raise ValueError(f"Invalid sentiment score: {sentiment_score_scalar}")
                
                # Update the rating based on sentiment
                updated_ratings[i, j] = (1 - alpha) * rating_matrix[i, j] + alpha * sentiment_score_scalar
    return updated_ratings


# Sample usage
# reviews = ["I love this product!", "It's okay.", "Quality is amazing, but delivery was late."]
# labels = np.array([1, 0, 1])  # Replace with actual labels
# tokenizer, bert_model = load_bert_model()

# bert_embeddings = get_bert_embeddings(reviews, tokenizer, bert_model)
# Initialize the SentimentRNN and compile
# sentiment_model = SentimentRNN()
# sentiment_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the sentiment model
# sentiment_model.fit(bert_embeddings.numpy(), labels, epochs=10, batch_size=2)

# Predict sentiment scores for new reviews
# sentiment_scores = get_sentiment_scores(reviews, tokenizer, bert_model)

# Adjust the ratings with sentiment scores
# updated_ratings = update_ratings_with_sentiment(rating_matrix, sentiment_scores)
