import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import itertools
import collections
import time
from statistics import mean
import pyclustering
from pyclustering.cluster import xmeans
import pandas as pd
from tqdm import tqdm
from amt_func import read_file, make_confusion_matrix, choice_teams, expectation, real_probability, distance_ranking

data_num = 50
worker_combi_num = 5
choices = list(range(0, 7))
df = pd.DataFrame()
breed_to_choice_dic = {'Coyo': 0, 'Dhol': 1, 'Husk': 2, 'Alas': 3, 'Samo': 4, 'Germ': 5, 'wolf': 6}
choice_num = len(breed_to_choice_dic)
delete_list = list(range(15)) + list(range(16, 28)) + list(range(48, 102))

#データ読み込み
#all_df, correct_answer_list = read_file(breed_to_choice_dic)
all_df = pd.read_pickle("all_df.pkl")
correct_answer_list = pd.read_pickle("correct_answer_list.pkl")

#vectorでの正当の次元
correct_answer_ref = []
for i in range(len(correct_answer_list)):
    correct_answer_ref.append(i*7 + correct_answer_list[i])

#事前確率
prior_probability = collections.Counter(correct_answer_list)
for key in prior_probability:
    prior_probability[key] = float(prior_probability[key]) / float(len(correct_answer_list))

#ワーカーの回答リスト
worker_answer_list = []
for index, row in all_df.iterrows():
    worker_answers = []
    for i, bool_value in enumerate(row.values.tolist()):
        if bool_value:
            worker_answers.append(i % choice_num)
    worker_answer_list.append(worker_answers)

#混同行列と足切り
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

#one-hotベクトル化
np_data = np.array(worker_answer_list).reshape(-1, 1)
enc = OneHotEncoder(categories="auto", sparse=False, dtype=np.float32)
one_hot_data = enc.fit_transform(np_data).reshape(data_num, -1)

answer_data_remove_correct = np.delete(one_hot_data, correct_answer_ref, 1)

correct_dim_twice_data = np.copy(one_hot_data)
correct_dim_twice_data[:, correct_answer_ref] = correct_dim_twice_data[:, correct_answer_ref] * 10000

#worker間の距離のランキング
sorted_distance_dic = distance_ranking(one_hot_data, data_num)


print('---------------------------------------------------')

list1 = []
list2 = []
list3 = []
list4 = []
for i in range(1):
    start = time.time()
    worker_combi_list = choice_teams(one_hot_data, data_num, worker_combi_num)
    elapsed_time = time.time() - start
    # print(worker_combi_list)
    expectation_list = []
    for worker_combi in worker_combi_list:
        confusion_matrix_combi = []
        for worker in worker_combi:
            confusion_matrix_combi.append(confusion_matrix_list[worker])
        result = expectation(confusion_matrix_combi, worker_combi_num, prior_probability)
        expectation_list.append(result)
    average_expectaion = mean(expectation_list)
    average_probability = real_probability(worker_combi_list, worker_answer_list, correct_answer_list)
    print("simple one-hot vector expectation value: " + str(average_expectaion))
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("average probability: " + str(average_probability))
    print('---------------------------------------------------')
    list1.append(average_expectaion)

    start = time.time()
    worker_combi_list = choice_teams(correct_dim_twice_data, data_num, worker_combi_num)
    elapsed_time = time.time() - start
    # print(worker_combi_list)
    expectation_list = []
    for worker_combi in worker_combi_list:
        confusion_matrix_combi = []
        for worker in worker_combi:
            confusion_matrix_combi.append(confusion_matrix_list[worker])
        result = expectation(confusion_matrix_combi, worker_combi_num, prior_probability)
        expectation_list.append(result)
    average_expectaion = mean(expectation_list)
    average_probability = real_probability(worker_combi_list, worker_answer_list, correct_answer_list)
    print("twice one-hot vector expectation value: " + str(average_expectaion))
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("average probability: " + str(average_probability))
    print('---------------------------------------------------')
    list2.append(average_expectaion)

    start = time.time()
    pred = KMeans(n_clusters=worker_combi_num).fit_predict(answer_data_remove_correct)
    worker_combi_list = choice_teams(answer_data_remove_correct, data_num, worker_combi_num)
    elapsed_time = time.time() - start
    expectation_list = []
    for worker_combi in worker_combi_list:
        confusion_matrix_combi = []
        for worker in worker_combi:
            confusion_matrix_combi.append(confusion_matrix_list[worker])
        result = expectation(confusion_matrix_combi, worker_combi_num, prior_probability)
        expectation_list.append(result)
    average_expectaion = mean(expectation_list)
    average_probability = real_probability(worker_combi_list, worker_answer_list, correct_answer_list)
    print("removed one-hot vector expectation value: " + str(average_expectaion))
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("average probability: " + str(average_probability))
    print('---------------------------------------------------')
    list3.append(average_expectaion)

    start = time.time()
    random_list = list(range(data_num))
    random.shuffle(random_list)
    random_combi_list = []
    for i in range(int(len(random_list) / worker_combi_num)):
        random_combi = []
        for j in range(worker_combi_num):
            random_combi.append(random_list[i * worker_combi_num + j])
        random_combi_list.append(random_combi)
    elapsed_time = time.time() - start
    expectation_list = []
    for worker_combi in random_combi_list:
        confusion_matrix_combi = []
        for worker in worker_combi:
            confusion_matrix_combi.append(confusion_matrix_list[worker])
        result = expectation(confusion_matrix_combi, worker_combi_num, prior_probability)
        expectation_list.append(result)
    average_expectaion = mean(expectation_list)
    average_probability = real_probability(random_combi_list, worker_answer_list, correct_answer_list)
    print("random selected worker expectation value: " + str(average_expectaion))
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("average probability: " + str(average_probability))
    print('---------------------------------------------------')
    list4.append(average_expectaion)

print(mean(list1))
print(mean(list2))
print(mean(list3))
print(mean(list4))
