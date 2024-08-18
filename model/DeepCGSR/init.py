import json
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../helper')))
from utils import read_data, word_segment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'review_processing')))
from fine_gain import DependencyParser
import nltk
nltk.download('wordnet')
nltk.download('sentiwordnet')

class config:
    dataset_name='reviewAmazon'
    user_length = 0
    item_length = 0
    data_path='model/DeepCGSR/feature/allFeatureReview.csv'
    data_feature='model/DeepCGSR/data/final_data_feature.csv'
    model_name='deepcgsr'
    k_topic = 10
    epoch=100
    learning_rate=0.01
    batch_size=32
    weight_decay=1e-6
    device='cuda:0'
    save_dir='chkpt'
    isRemoveOutliner = False

args = config()

#set parameters for review processing
model_path = 'model/DeepCGSR/config/stanford-corenlp-4.5.7/stanford-corenlp-4.5.7-models.jar'
parser_path = 'model/DeepCGSR/config/stanford-corenlp-4.5.7/stanford-corenlp-4.5.7.jar'
dep_parser = DependencyParser(model_path, parser_path)