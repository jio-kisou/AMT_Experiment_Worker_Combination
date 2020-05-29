import numpy as np
from amt_func import read_file, make_confusion_matrix
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AffinityPropagation


data_num = 40
breed_to_choice_dic = {'Coyo': 0, 'Dhol': 1, 'Husk': 2, 'Alas': 3, 'Samo': 4, 'Germ': 5, 'wolf': 6}
choice_num = len(breed_to_choice_dic)


all_df = pd.read_pickle("all_df.pkl")
correct_answer_list = pd.read_pickle("correct_answer_list.pkl")

# vectorでの正当の次元
correct_answer_ref = []
for i in range(len(correct_answer_list)):
    correct_answer_ref.append(i*7 + correct_answer_list[i])

# ワーカーの回答リスト
worker_answer_list = []
for index, row in all_df.iterrows():
    worker_answers = []
    for i, bool_value in enumerate(row.values.tolist()):
        if bool_value:
            worker_answers.append(i % choice_num)
    worker_answer_list.append(worker_answers)


# 混同行列と足切り
confusion_matrix_list, accurate_list = make_confusion_matrix(correct_answer_list, worker_answer_list, choice_num)
sorted_accurate_list = sorted(enumerate(accurate_list), key=lambda x: x[1])
low_accurate_list = []
for i in range(len(confusion_matrix_list) - data_num):
    low_accurate_list.append(sorted_accurate_list[i][0])
print("cutted worker num: " + str(len(low_accurate_list)))
for i in sorted(low_accurate_list, reverse=True):
    confusion_matrix_list.pop(i)
for i in sorted(low_accurate_list, reverse=True):
    worker_answer_list.pop(i)

# one-hotベクトル化
np_data = np.array(worker_answer_list).reshape(-1, 1)
enc = OneHotEncoder(categories="auto", sparse=False, dtype=np.float32)
one_hot_data = enc.fit_transform(np_data).reshape(data_num, -1)
answer_data_remove_correct = np.delete(one_hot_data, correct_answer_ref, 1)


# Affinity Propagation
distance_matrix = np.zeros((data_num, data_num))
for i in range(data_num):
    for j in range(data_num):
        distance_i_j = 0
        for dim_num in range(answer_data_remove_correct.shape[1]):
            if answer_data_remove_correct[i, dim_num] == answer_data_remove_correct[j, dim_num]:
                distance_i_j = distance_i_j + 1
        distance_matrix[i, j] = distance_i_j
        distance_matrix[j, i] = distance_i_j

affinity_propagation = AffinityPropagation(affinity='precomputed')
f = affinity_propagation.fit(distance_matrix)
print(f.labels_)
