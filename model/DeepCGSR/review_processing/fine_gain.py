
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import sentiwordnet as swn
from nltk.parse.stanford import StanfordDependencyParser
from sklearn.cluster import KMeans, Birch, DBSCAN, MeanShift, BisectingKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score
from transformers import BertTokenizer, BertModel
import torch
import sys
import os
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from helper.utils import preprocessed, word_segment


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',  # Tự động padding các chuỗi văn bản ngắn hơn max_len
            truncation=True,  # Cắt bớt các chuỗi dài hơn max_len
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(tokenizer):
    def collate_batch(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Pad input_ids và attention_mask đến kích thước của chuỗi dài nhất trong batch
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    return collate_batch


def fine_tune_bert(texts, labels, num_labels, epochs=50, batch_size=8, max_len=512, learning_rate=2e-5, save_dir='./model/DeepCGSR/chkpt'):
    # Kiểm tra nếu GPU có sẵn và thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Khởi tạo tokenizer và dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CustomDataset(texts, labels, tokenizer, max_len)

    # DataLoader với collate_fn
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn(tokenizer))

    # Load pre-trained BERT và optimizer, chuyển mô hình sang GPU (nếu có)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model = model.to(device)  # Chuyển mô hình sang GPU

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    start_epoch = 0
    checkpoint_path = os.path.join(save_dir, "bert_last_checkpoint.pt")

    # Kiểm tra xem checkpoint có tồn tại không
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}. Loading checkpoint.")
        checkpoint = torch.load(checkpoint_path, map_location=device)  # Load checkpoint vào GPU (nếu có)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch + 1}")
        return model, tokenizer

    # Training loop
    for epoch in range(start_epoch, epochs):  # Bắt đầu từ epoch đã lưu
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Chuyển các tensor của batch sang GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"Loss": total_loss / len(progress_bar)})

        # Lưu mô hình sau mỗi epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(loader)
        }, checkpoint_path)
        
        print(f"Epoch {epoch + 1} complete. Model saved to {checkpoint_path}.")

    print(f"Training complete. Final loss: {total_loss / len(loader):.4f}")

    return model, tokenizer

def get_bert_embeddings(texts, tokenizer, model, device):
    # Tokenize văn bản và chuyển các tensor sang GPU (nếu có)
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.bert(**inputs)
    
    # Lấy embedding cuối cùng và trung bình trên chiều thứ 1
    return outputs.last_hidden_state.mean(dim=1)

# Load pre-trained BERT model and tokenizer
def get_tbert_model(data_df, split_data, num_topics, num_words, cluster_method='Kmeans'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cleaned_data = data_df.dropna(subset=['filteredReviewText', 'overall_new'])
    cleaned_data['overall_new'] = cleaned_data['overall_new'].apply(lambda x: 1 if x > 3 else 0)

    texts = cleaned_data['filteredReviewText'].tolist()
    labels = cleaned_data['overall_new'].tolist()

    # Fine-tune BERT và chuyển mô hình sang GPU
    model, tokenizer = fine_tune_bert(texts, labels, num_labels=5, epochs=10)
    model = model.to(device)
    
    # Lấy embeddings từ mô hình BERT
    embeddings = []
    for text in split_data:
        embedding = get_bert_embeddings([' '.join(text)], tokenizer, model, device)
        embeddings.append(embedding)

    embeddings = torch.vstack(embeddings).to(device)

    # Clustering để tìm chủ đề
    if cluster_method == 'Kmeans':
        print("Kmeans")
        clustering = KMeans(n_clusters=num_topics, random_state=42).fit(embeddings.cpu().numpy())
    elif cluster_method == 'Birch':
        print("Birch")
        clustering = Birch(n_clusters=num_topics).fit(embeddings.cpu().numpy())
    elif cluster_method == 'DBSCAN':
        print("DBSCAN")
        clustering = DBSCAN(eps=3, min_samples=num_topics).fit(embeddings.cpu().numpy())
    elif cluster_method == 'MeanShift':
        print("MeanShift")
        clustering = MeanShift(bandwidth=num_topics).fit(embeddings.cpu().numpy())
    elif cluster_method == 'BisectingKMeans':
        print("BisectingKMeans")
        clustering = BisectingKMeans(n_clusters=num_topics, random_state=0).fit(embeddings.cpu().numpy())

    labels = clustering.labels_
    
    # Tính silhouette score
    silhouette_score_val = silhouette_score(embeddings.cpu().numpy(), labels)
    
    evaluation_df = pd.DataFrame({
        'num_topics': [num_topics],
        'silhouette_score_val': [silhouette_score_val]
    })

    # Lưu vào file CSV
    file_path = 'model/DeepCGSR/evaluation_clustering/silhouette_score.csv'
    evaluation_df.to_csv(file_path, index=False)

    # Tiếp tục xử lý văn bản và trích xuất từ top của mỗi chủ đề...
    topic_to_words = []
    for i in range(num_topics):
        cluster_indices = [j for j, label in enumerate(labels) if label == i]
        cluster_texts = [' '.join(split_data[j]) for j in cluster_indices]
        cluster_texts = [text for text in cluster_texts if text.strip()]

        # Tránh lỗi nếu không có tài liệu hợp lệ
        if not cluster_texts:
            topic_to_words.append([])
            continue

        # Vector hóa văn bản
        vectorizer = TfidfVectorizer(max_features=num_words, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)

        indices = np.argsort(tfidf_matrix.sum(axis=0)).flatten()[::-1]
        feature_names = vectorizer.get_feature_names_out()
        top_words = [feature_names[ind] for ind in indices[:num_words]]
        topic_to_words.append(top_words)
    
    # Tạo dictionary cho tài liệu
    dictionary = corpora.Dictionary(split_data)
    
    return embeddings, model, clustering, dictionary, tokenizer, topic_to_words


def get_lda_model(split_data, num_topics, num_words):
    dictionary = corpora.Dictionary(split_data)
    corpus = [dictionary.doc2bow(text) for text in split_data]

    # LDA模型训练
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

    # 主题单词矩阵
    topic_to_words = []
    for i in range(num_topics):
        cur_topic_words = [ele[0] for ele in model.show_topic(i, num_words)]
        topic_to_words.append(cur_topic_words)
    return model, dictionary, topic_to_words


# Get sentiment score for each topic
class DependencyParser():
    def __init__(self, model_path, parser_path):
        self.model = StanfordDependencyParser(path_to_jar=parser_path, path_to_models_jar=model_path)

    def raw_parse(self, text):
        parse_result = self.model.raw_parse(text)
        result = [list(parse.triples()) for parse in parse_result]
        return result[0]

def get_word_sentiment_score(word):
    m = list(swn.senti_synsets(word, "n"))
    s = 0
    for j in range(len(m)):
        s += (m[j].pos_score() - m[j].neg_score())
    return s

def get_topic_sentiment_metrix_lda(text, dictionary, lda_model, topic_word_metrix, dependency_parser, topic_nums=50):
    """获取主题-情感矩阵
    """
    text_p = word_segment(text)
    doc_bow = dictionary.doc2bow(text_p)  # 文档转换成bow
    doc_lda = lda_model[doc_bow]  # [(12, 0.042477883), (13, 0.36870235), (16, 0.35455772), (37, 0.20635633)]

    # 初始化主题矩阵
    topci_sentiment_m = np.zeros(topic_nums)

    # 获取依存句法分析结果
    sentences = preprocessed(text)
    dep_parser_result_p = []
    for i in sentences:
        # 依存句法分析
        # print(i)
        dep_parser_result = dependency_parser.raw_parse(i)
        # print(dep_parser_result)
        for j in dep_parser_result:
            dep_parser_result_p.append([j[0][0], j[2][0]])
    #     print(dep_parser_result_p)
    # print(doc_lda)
    for topic_id, _ in doc_lda:
        # 获取当前主题的特征词
        cur_topic_words = topic_word_metrix[topic_id]
        cur_topic_sentiment = 0
        cur_topci_senti_word = []

        # 根据特征词获取情感词
        # print("当前句子", word_segment(text))
        for word in word_segment(text):
            # 获取当前文本出现的特征词
            if word in cur_topic_words:
                cur_topci_senti_word.append(word)
                # 根据依存关系， 获得依存词。 并计算主题情感
                for p in dep_parser_result_p:
                    if p[0] == word:
                        # 将依存词的情感加入主题
                        cur_topci_senti_word.append(p[1])
                    if p[1] == word:
                        cur_topci_senti_word.append(p[0])

        for senti_word in cur_topci_senti_word:
            # cur_topic_sentiment += word_to_senti.get(senti_word, 0)
            cur_topic_sentiment += get_word_sentiment_score(senti_word)
        # print("cur_topci_senti_word", cur_topci_senti_word)
        # 主题情感取值范围[-5, 5]
        if cur_topic_sentiment > 5:
            cur_topic_sentiment = 5
        elif cur_topic_sentiment < -5:
            cur_topic_sentiment = -5

        topci_sentiment_m[topic_id] = cur_topic_sentiment
    return topci_sentiment_m
#=================================================
import torch
import numpy as np
from nltk.corpus import wordnet as wn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Sử dụng GPU nếu có

def get_word_sentiment_score_by_vader(word):
    sentiment_dict = analyzer.polarity_scores(word)
    return sentiment_dict['compound']

def get_top_synonyms(word, top_n=4):
    noun_synsets = wn.synsets(word, pos=wn.NOUN)
    adj_synsets = wn.synsets(word, pos=wn.ADJ)
    
    all_synsets = noun_synsets + adj_synsets
    synonym_scores = []
    for synset in all_synsets:
        for lemma in synset.lemma_names():
            if lemma.lower() != word.lower() and lemma not in [syn[0] for syn in synonym_scores]:
                synonym_scores.append((lemma))
    
    return synonym_scores[:top_n]

def get_word_sentiment_score_addition(word):
    m = list(swn.senti_synsets(word))
    s = 0
    if not m:
        return s  # Trả về 0 nếu không tìm thấy synset nào cho từ này
    for synset in m:
        if synset.pos_score() == 0 and synset.neg_score() == 0:
            s += get_word_sentiment_score_by_vader(synset.synset.name().split('.')[0])
        else:
            s += (synset.pos_score() - synset.neg_score())
    return s

def get_synonyms_sentiment_scores(word, top_n=4):
    synonyms = get_top_synonyms(word, top_n=top_n)
    scores = 0
    
    for synonym in synonyms:
        sentiment_score = get_word_sentiment_score_addition(synonym)
        scores += sentiment_score

    scores = scores / top_n
    return scores

def get_topic_sentiment_matrix_tbert(text, topic_word_matrix, dependency_parser, topic_nums=50):
    topic_sentiment_m = torch.zeros(topic_nums, device=device)  # Đảm bảo ma trận sentiment ở trên GPU

    try:
        sentences = preprocessed(text)
        dep_parser_result_p = []
        
        for i in sentences:
            dep_parser_result = dependency_parser.raw_parse(i)
            for j in dep_parser_result:
                dep_parser_result_p.append([j[0][0], j[2][0]])
                
        for topic_id, cur_topic_words in enumerate(topic_word_matrix):
            cur_topic_senti_word = []
            for word in word_segment(text):
                if any(word in sublist for sublist in cur_topic_words):
                    cur_topic_senti_word.append(word)
                    for p in dep_parser_result_p:
                        if p[0] == word:
                            cur_topic_senti_word.append(p[1])
                        if p[1] == word:
                            cur_topic_senti_word.append(p[0])

            if cur_topic_senti_word:  # Kiểm tra nếu danh sách không rỗng
                cur_topic_sentiment = sum(get_synonyms_sentiment_scores(senti_word) for senti_word in cur_topic_senti_word)
                topic_sentiment_m[topic_id] = torch.tensor(np.clip(cur_topic_sentiment, -5, 5), device=device)  # Chuyển sang GPU
            else:
                topic_sentiment_m[topic_id] = torch.tensor(0, device=device)  # Chuyển giá trị mặc định sang GPU
                
        return topic_sentiment_m
    except Exception as e:
        print("get_topic_sentiment_matrix_tbert's error: ", e, " text: ", text)
        return topic_sentiment_m



