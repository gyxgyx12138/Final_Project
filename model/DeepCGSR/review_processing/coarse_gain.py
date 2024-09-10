from model.DeepCGSR.review_processing.fine_gain import get_word_sentiment_score_addition, get_word_sentiment_score
from helper.utils import softmax, word_segment, sigmoid
from gensim.models import word2vec, Word2Vec



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

