
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../helper')))
from helper.general_functions import create_and_write_csv, read_csv_file
from model.DeepCGSR.review_processing.merge_senmatic_review import reviewer_feature_dict, item_feature_dict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, Concatenate, Dot, Add


#============================ Calulate U/I deep ===============================

def Calculate_Deep(v, z):
    v_z = v * z
    v2_z2 = (v**2) * (z**2)
    result = (1 / 2) * ((v_z)**2 - v2_z2)
    return result

def Calculate_Deep_Orginal(x, v):
    k = v.size  # Số lượng đặc trưng
    deep = np.zeros(k)  # Khởi tạo vector k-chiều cho các đặc trưng sâu
    sum_vx = np.dot(v, x)
    sum_vx_square = np.sum(v**2 * x**2)
    deep = 0.5 * (sum_vx**2 - sum_vx_square)

    return deep

def mergeReview_Rating(path, filename, svd, reviewer_feature_dict, item_feature_dict, getEmbedding, method_name):
    reviewerID,_ = read_csv_file(path)
    feature_dict = {}
    review_feature_list = []
    rating_feature_list = []
    for id in reviewerID:
        if getEmbedding == "reviewer":
            A = reviewer_feature_dict[id]
            B = svd.get_user_embedding(id)
        else:
            A = item_feature_dict[id]
            B = svd.get_item_embedding(id)

        z = np.concatenate((np.array(A), np.array(B)))
        feature_dict[id] = z
        review_feature_list.append(A)
        rating_feature_list.append(B)
    create_and_write_csv(filename, feature_dict, method_name)
    return feature_dict, review_feature_list, rating_feature_list

def merge_features(review_feature, rating_feature, num_factors, model_type='DCN'):

    input_text = Input(shape=(len(review_feature),))
    input_rating = Input(shape=(len(rating_feature),))

    if model_type == 'DCN':
        deep = Dense(num_factors, activation='relu')(input_text)
        deep = Dense(num_factors, activation='relu')(deep)

        cross = Multiply()([input_text, input_rating])
        cross = Add()([cross, input_text])
        merged = Concatenate()([deep, cross])

    elif model_type == 'NCF':
        text_embedding = Dense(num_factors, activation='relu')(input_text)
        rating_embedding = Dense(num_factors, activation='relu')(input_rating)

        gmf = Multiply()([text_embedding, rating_embedding])

        mlp = Concatenate()([text_embedding, rating_embedding])
        mlp = Dense(num_factors, activation='relu')(mlp)
        merged = Concatenate()([gmf, mlp])

    else:
        raise ValueError("model_type must be either 'DCN' or 'NCF'")

    output = Dense(len(review_feature), activation='relu')(merged)
    model = Model(inputs=[input_text, input_rating], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    merged_vector = model.predict([np.array([review_feature]), np.array([rating_feature])])

    return merged_vector.flatten()

def merge_features_mf(review_feature, rating_feature, num_factors):
    # Input layers
    input_text = Input(shape=(len(review_feature),))
    input_rating = Input(shape=(len(rating_feature),))

    # Embedding layers for review and rating features
    review_embedding = Dense(num_factors, activation='relu')(input_text)
    rating_embedding = Dense(num_factors, activation='relu')(input_rating)
    merged = Dot(axes=1)([review_embedding, rating_embedding])
    output = Dense(len(review_feature), activation='relu')(merged)

    # Model
    model = Model(inputs=[input_text, input_rating], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    merged_vector = model.predict([np.array([review_feature]), np.array([rating_feature])])

    return merged_vector.flatten()



