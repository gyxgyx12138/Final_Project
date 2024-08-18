
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../helper')))
from general_functions import create_and_write_csv, read_csv_file
from model.DeepCGSR.review_processing.merge_senmatic_review import reviewer_feature_dict, item_feature_dict


#============================ Calulate U/I deep ===============================

def Calculate_Deep(v, z, start):
    list_sum = {}
    i = start
    for name, z_i in z.items():
        if i < len(v):  # Đảm bảo vẫn còn phần tử trong danh sách v
            v_i = v[i]
            v_i = np.array(v_i)
            v_z = v_i * z_i
            v2_z2 = (v_i**2) * (z_i**2)
            result = (1 / 2) * ((v_z)**2 - v2_z2)
            list_sum[name] = result
            i += 1
    return list_sum

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
    # z_list = []
    # v_list = []
    for id in reviewerID:
        if getEmbedding == "reviewer":
            A = reviewer_feature_dict[id]
            B = svd.get_user_embedding(id)
        else:
            A = item_feature_dict[id]
            B = svd.get_item_embedding(id)

        z = np.concatenate((np.array(A), np.array(B)))
        feature_dict[id] = z
        # z_list.append(A)
        # v_list.append(B)
    create_and_write_csv(filename, feature_dict, method_name)
    return feature_dict




