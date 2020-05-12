import csv
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
#from sklearn_extra.cluster import KMedoids
import itertools
import collections
import time
from statistics import mean
import pyclustering
from pyclustering.cluster import xmeans



data_num = 100
worker_combi_num = 5
choices = list(range(0, 5))
choice_num = len(choices)
answer_data = []
# prob1 = [0.7, 0.075, 0.075, 0.075, 0.075]
# for i in range(data_num):
#     answer_list = []
#     for j in choices:
#         answer_j = np.random.choice(a=choices, size=20, p=prob1)
#         answer_list = answer_list + answer_j.tolist()
#         prob1 = np.roll(np.array(prob1), 1).tolist()
#     answer_data.append(answer_list)

prob_1 = [0.7, 0.3, 0.0, 0.0, 0.0]
for i in range(25):
    answer_list = []
    for j in choices:
        answer_j = np.random.choice(a=choices, size=20, p=prob_1)
        answer_list = answer_list + answer_j.tolist()
        prob_1 = np.roll(np.array(prob_1), 1).tolist()
    answer_data.append(answer_list)

prob_2 = [0.7, 0.0, 0.3, 0.0, 0.0]
for i in range(25):
    answer_list = []
    for j in choices:
        answer_j = np.random.choice(a=choices, size=20, p=prob_2)
        answer_list = answer_list + answer_j.tolist()
        prob_2 = np.roll(np.array(prob_2), 1).tolist()
    answer_data.append(answer_list)

prob_3 = [0.7, 0.0, 0.0, 0.3, 0.0]
for i in range(25):
    answer_list = []
    for j in choices:
        answer_j = np.random.choice(a=choices, size=20, p=prob_3)
        answer_list = answer_list + answer_j.tolist()
        prob_3 = np.roll(np.array(prob_3), 1).tolist()
    answer_data.append(answer_list)

prob_4 = [0.7, 0.0, 0.0, 0.0, 0.3]
for i in range(25):
    answer_list = []
    for j in choices:
        answer_j = np.random.choice(a=choices, size=20, p=prob_4)
        answer_list = answer_list + answer_j.tolist()
        prob_4 = np.roll(np.array(prob_4), 1).tolist()
    answer_data.append(answer_list)

# prob_1 = [0.6, 0.4, 0.0, 0.0, 0.0]
# for i in range(25):
#     answer_list = []
#     for j in choices:
#         answer_j = np.random.choice(a=choices, size=20, p=prob_1)
#         answer_list = answer_list + answer_j.tolist()
#         prob_1 = np.roll(np.array(prob_1), 1).tolist()
#     answer_data.append(answer_list)
#
# prob_2 = [0.6, 0.0, 0.4, 0.0, 0.0]
# for i in range(25):
#     answer_list = []
#     for j in choices:
#         answer_j = np.random.choice(a=choices, size=20, p=prob_2)
#         answer_list = answer_list + answer_j.tolist()
#         prob_2 = np.roll(np.array(prob_2), 1).tolist()
#     answer_data.append(answer_list)
#
# prob_3 = [0.6, 0.0, 0.0, 0.4, 0.0]
# for i in range(25):
#     answer_list = []
#     for j in choices:
#         answer_j = np.random.choice(a=choices, size=20, p=prob_3)
#         answer_list = answer_list + answer_j.tolist()
#         prob_3 = np.roll(np.array(prob_3), 1).tolist()
#     answer_data.append(answer_list)
#
# prob_4 = [0.6, 0.0, 0.0, 0.0, 0.4]
# for i in range(25):
#     answer_list = []
#     for j in choices:
#         answer_j = np.random.choice(a=choices, size=20, p=prob_4)
#         answer_list = answer_list + answer_j.tolist()
#         prob_4 = np.roll(np.array(prob_4), 1).tolist()
#     answer_data.append(answer_list)

correct_answer_list = []
for i in range(len(answer_data[0])):
    if i < 20:
        correct_answer_list.append(0)
    elif i < 40:
        correct_answer_list.append(1)
    elif i < 60:
        correct_answer_list.append(2)
    elif i < 80:
        correct_answer_list.append(3)
    else:
        correct_answer_list.append(4)

accurate_dic = {}
for j, answer_list in enumerate(answer_data):
    correct_num = 0
    for i in range(100):
        if answer_list[i] == correct_answer_list[i]:
            correct_num = correct_num + 1
    accurate_dic[j] = float(correct_num)/100.0

confusion_matrix_dic = {}
for i, answer_list in enumerate(answer_data):
    num00 = 0
    num01 = 0
    num02 = 0
    num03 = 0
    num04 = 0
    num10 = 0
    num11 = 0
    num12 = 0
    num13 = 0
    num14 = 0
    num20 = 0
    num21 = 0
    num22 = 0
    num23 = 0
    num24 = 0
    num30 = 0
    num31 = 0
    num32 = 0
    num33 = 0
    num34 = 0
    num40 = 0
    num41 = 0
    num42 = 0
    num43 = 0
    num44 = 0
    for j in range(100):
        if j < 20:
            if answer_list[j] == 0:
                num00 = num00 + 1
            elif answer_list[j] == 1:
                num01 = num01 + 1
            elif answer_list[j] == 2:
                num02 = num02 + 1
            elif answer_list[j] == 3:
                num03 = num03 + 1
            else:
                num04 = num04 + 1
        elif j < 40:
            if answer_list[j] == 0:
                num10 = num10 + 1
            elif answer_list[j] == 1:
                num11 = num11 + 1
            elif answer_list[j] == 2:
                num12 = num12 + 1
            elif answer_list[j] == 3:
                num13 = num13 + 1
            else:
                num14 = num14 + 1
        elif j < 60:
            if answer_list[j] == 0:
                num20 = num20 + 1
            elif answer_list[j] == 1:
                num21 = num21 + 1
            elif answer_list[j] == 2:
                num22 = num22 + 1
            elif answer_list[j] == 3:
                num23 = num23 + 1
            else:
                num24 = num24 + 1
        elif j < 80:
            if answer_list[j] == 0:
                num30 = num30 + 1
            elif answer_list[j] == 1:
                num31 = num31 + 1
            elif answer_list[j] == 2:
                num32 = num32 + 1
            elif answer_list[j] == 3:
                num33 = num33 + 1
            else:
                num34 = num34 + 1
        else:
            if answer_list[j] == 0:
                num40 = num40 + 1
            elif answer_list[j] == 1:
                num41 = num41 + 1
            elif answer_list[j] == 2:
                num42 = num42 + 1
            elif answer_list[j] == 3:
                num43 = num43 + 1
            else:
                num44 = num44 + 1
    confusion_matrix = np.array([[float(num00)/20.0, float(num01)/20.0, float(num02)/20.0, float(num03)/20.0, float(num04)/20.0],
                                 [float(num10)/20.0, float(num11)/20.0, float(num12)/20.0, float(num13)/20.0, float(num14)/20.0],
                                 [float(num20)/20.0, float(num21)/20.0, float(num22)/20.0, float(num23)/20.0, float(num24)/20.0],
                                 [float(num30)/20.0, float(num31)/20.0, float(num32)/20.0, float(num33)/20.0, float(num34)/20.0],
                                 [float(num40)/20.0, float(num41)/20.0, float(num42)/20.0, float(num43)/20.0, float(num44)/20.0]])
    confusion_matrix_dic[i] = confusion_matrix

correct_answer_ref = []
for i in range(len(answer_data[0])):
    if i < 20:
        correct_answer_ref.append(i*5)
    elif i < 40:
        correct_answer_ref.append(i*5+1)
    elif i < 60:
        correct_answer_ref.append(i*5+2)
    elif i < 80:
        correct_answer_ref.append(i*5+3)
    else:
        correct_answer_ref.append(i*5+4)


def pai_multi(data, correct_answer, answers):
    c = collections.Counter(answers)
    majority = c.most_common()[0][0]
    result = 1
    for j in range(len(data)):
        result = result * data[j][majority, answers[j]]
    return result


def expectation(combi):
    sum = 0
    arr = []
    for i, j, k, l, m in itertools.product(choices, choices, choices, choices, choices):
        answers = [i, j, k, l, m]
        a = pai_multi(combi, choices, answers)
        sum = sum + a/float(choice_num)
        arr.append(a/float(choice_num))
    return sum, arr


start = time.time()
top_accutate_combi = sorted(accurate_dic.items(), key=lambda x:x[1], reverse=True)
top_combi_list = []
for i in range(int(len(top_accutate_combi)/5)):
    top_combi_list.append([top_accutate_combi[i][0], top_accutate_combi[i+20][0], top_accutate_combi[i+40][0],
                           top_accutate_combi[i+60][0], top_accutate_combi[i+80][0]])
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
#print(top_accutate_combi)
expectation_list = []
for i in range(len(top_combi_list)):
    result = expectation([confusion_matrix_dic[top_combi_list[i][0]],
                          confusion_matrix_dic[top_combi_list[i][1]],
                          confusion_matrix_dic[top_combi_list[i][2]],
                          confusion_matrix_dic[top_combi_list[i][3]],
                          confusion_matrix_dic[top_combi_list[i][4]]])
    expectation_list.append(result[0])
average_expectaion = mean(expectation_list)
print(average_expectaion)


start = time.time()
list = list(range(data_num))
random.shuffle(list)
random_combi_list = []
for i in range(int(len(list)/5)):
    random_combi_list.append([list[i*5], list[i*5+1], list[i*5+2], list[i*5+3], list[i*5+4]])
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
expectation_list = []
for i in range(len(random_combi_list)):
    result = expectation([confusion_matrix_dic[random_combi_list[i][0]],
                          confusion_matrix_dic[random_combi_list[i][1]],
                          confusion_matrix_dic[random_combi_list[i][2]],
                          confusion_matrix_dic[random_combi_list[i][3]],
                          confusion_matrix_dic[random_combi_list[i][4]]])
    expectation_list.append(result[0])
average_expectaion = mean(expectation_list)
print(average_expectaion)

# half_data_num = int(data_num/2)
# worse_accurate_combi = top_accutate_combi[half_data_num:]
# delete_list = []
# for i in worse_accurate_combi:
#     delete_list.append(i[0])
#
# answer_tophalf_data = np.delete(np.array(answer_data), delete_list, 0)
np_data = np.array(answer_data).reshape(-1, 1)
enc = OneHotEncoder(categories="auto", sparse=False, dtype=np.float32)
one_hot_data = enc.fit_transform(np_data).reshape(data_num, -1)

answer_data_remove_correct = np.delete(one_hot_data, correct_answer_ref, 1)

# twice_one_hot_data = 0
# for i in range(100):
#     if i < 20:

#print(answer_data_remove_correct.shape)


def better_combi(pred):
    cluster0 = {}
    cluster1 = {}
    cluster2 = {}
    cluster3 = {}
    cluster4 = {}
    for i, cluster in enumerate(pred.tolist()):
        if cluster == 0:
            cluster0[i] = accurate_dic[i]
        elif cluster == 1:
            cluster1[i] = accurate_dic[i]
        elif cluster == 2:
            cluster2[i] = accurate_dic[i]
        elif cluster == 3:
            cluster3[i] = accurate_dic[i]
        elif cluster == 4:
            cluster4[i] = accurate_dic[i]
    max0 = max(cluster0, key=cluster0.get)
    max1 = max(cluster1, key=cluster1.get)
    max2 = max(cluster2, key=cluster2.get)
    max3 = max(cluster3, key=cluster3.get)
    max4 = max(cluster4, key=cluster4.get)
    return [max0, max1, max2, max3, max4]


def random_combi(pred, data):
    cluster_list = [[], [], [], [], []]
    for i, cluster in enumerate(pred.tolist()):
        if cluster == 0:
            cluster_list[0].append(i)
        elif cluster == 1:
            cluster_list[1].append(i)
        elif cluster == 2:
            cluster_list[2].append(i)
        elif cluster == 3:
            cluster_list[3].append(i)
        elif cluster == 4:
            cluster_list[4].append(i)
    for i in range(len(cluster_list)):
        random.shuffle(cluster_list[i])
    data_ref_list = [num for num in range(100)]
    worker_combi_list = []
    i = 0
    while len(worker_combi_list) < data_num / worker_combi_num:
        worker_combi = []
        worker_combi_connect = []
        if len(cluster_list[0]) == 0 or len(cluster_list[1]) == 0 or len(cluster_list[2]) == 0 or len(cluster_list[3]) == 0 or len(cluster_list[4]) == 0:
            data = np.delete(data, worker_combi_connect, 0)
            data_ref_list = np.delete(np.array(data_ref_list), worker_combi_connect).tolist()
            pred_kai = KMeans(n_clusters=worker_combi_num).fit_predict(data)
            cluster_list = [[], [], [], [], []]
            for i, cluster in enumerate(pred_kai.tolist()):
                if cluster == 0:
                    cluster_list[0].append(i)
                elif cluster == 1:
                    cluster_list[1].append(i)
                elif cluster == 2:
                    cluster_list[2].append(i)
                elif cluster == 3:
                    cluster_list[3].append(i)
                elif cluster == 4:
                    cluster_list[4].append(i)
            for i in range(len(cluster_list)):
                random.shuffle(cluster_list[i])
        for i in range(len(cluster_list)):
            worker_combi.append(data_ref_list[cluster_list[i][0]])
            del cluster_list[i][0]
        worker_combi_connect = worker_combi_connect + worker_combi
        worker_combi_list.append(worker_combi)
    return worker_combi_list


start = time.time()
pred = KMeans(n_clusters=worker_combi_num).fit_predict(one_hot_data)
worker_combi_list = random_combi(pred, one_hot_data)
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
expectation_list = []
for i in range(len(worker_combi_list)):
    result = expectation([confusion_matrix_dic[worker_combi_list[i][0]],
                          confusion_matrix_dic[worker_combi_list[i][1]],
                          confusion_matrix_dic[worker_combi_list[i][2]],
                          confusion_matrix_dic[worker_combi_list[i][3]],
                          confusion_matrix_dic[worker_combi_list[i][4]]])
    expectation_list.append(result[0])
average_expectaion = mean(expectation_list)
print(average_expectaion)

init_center = pyclustering.cluster.xmeans.kmeans_plusplus_initializer(one_hot_data, 4).initialize() # 初期値決定　今回は、初期クラスタ2です
xm = pyclustering.cluster.xmeans.xmeans(one_hot_data, init_center, ccore=False)
xm.process()
clusters = xm.get_clusters()


start = time.time()
pred2 = KMeans(n_clusters=worker_combi_num).fit_predict(answer_data_remove_correct)
worker_combi_list2 = random_combi(pred2, answer_data_remove_correct)
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
expectation_list2 = []
for i in range(len(worker_combi_list2)):
    result = expectation([confusion_matrix_dic[worker_combi_list2[i][0]],
                          confusion_matrix_dic[worker_combi_list2[i][1]],
                          confusion_matrix_dic[worker_combi_list2[i][2]],
                          confusion_matrix_dic[worker_combi_list2[i][3]],
                          confusion_matrix_dic[worker_combi_list2[i][4]]])
    expectation_list2.append(result[0])
average_expectaion2 = mean(expectation_list2)
print(average_expectaion2)
print(average_expectaion2)


# kmeans_combi = better_combi(pred)
# print(kmeans_combi)
# start = time.time()
# result3 = expectation([confusion_matrix_dic[kmeans_combi[0]],
#                        confusion_matrix_dic[kmeans_combi[1]],
#                        confusion_matrix_dic[kmeans_combi[2]],
#                        confusion_matrix_dic[kmeans_combi[3]],
#                        confusion_matrix_dic[kmeans_combi[4]]])
# elapsed_time = time.time() - start
# print(round(result3[0], 5))
# print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


# pred2 = KMeans(n_clusters=worker_combi_num).fit_predict(answer_data_remove_correct)
# kmeans_combi2 = better_combi(pred2)
# print(kmeans_combi2)
#
# result3 = expectation([confusion_matrix_dic[kmeans_combi2[0]],
#                        confusion_matrix_dic[kmeans_combi2[1]],
#                        confusion_matrix_dic[kmeans_combi2[2]],
#                        confusion_matrix_dic[kmeans_combi2[3]],
#                        confusion_matrix_dic[kmeans_combi2[4]]])
# print(round(result3[0], 5))
#
#
# combination = list(itertools.combinations(range(len(confusion_matrix_dic)), worker_combi_num))
# def best_combi(data):
#     expect_list = []
#     expect_dic = {}
#     for combi in combination:
#         target_combi = [data[combi[0]], data[combi[1]], data[combi[2]], data[combi[3]], data[combi[4]]]
#         expect = expectation(target_combi)[0]
#         expect_list.append(expect)
#         expect_dic[i] = expect
#     print(np.amax(expect_list))
#     return combination[np.argmax(expect_list)], expect_dic
#
# best = best_combi(confusion_matrix_dic)
# print(best[0])
#
#
#
# #kmedoids = KMedoids(n_clusters=10, metric='hamming', random_state=0).fit_predict(one_hot_data)
# #print(kmedoids)
#
# def random_combi(pred):
#     cluster_list = [[], [], [], [], []]
#     for i, cluster in enumerate(pred.tolist()):
#         if cluster == 0:
#             cluster_list[0].append(i)
#         elif cluster == 1:
#             cluster_list[1].append(i)
#         elif cluster == 2:
#             cluster_list[2].append(i)
#         elif cluster == 3:
#             cluster_list[3].append(i)
#         elif cluster == 4:
#             cluster_list[4].append(i)
#     for i in range(len(cluster_list)):
#         random.shuffle(cluster_list[i])
#     size_cluster = {}
#     for i in range(len(cluster_list)):
#         size_cluster[i] = len(cluster_list[i])
#     worker_combi_list = []
#     i = 0
#     while len(worker_combi_list) < data_num / worker_combi_num:
#         worker_combi = []
#         size_cluster.clear()
#         for i in range(len(cluster_list)):
#             if len(cluster_list[i]) != 0:
#                 worker_combi.append(cluster_list[i][0])
#                 del cluster_list[i][0]
#         for i in range(len(cluster_list)):
#             if len(cluster_list[i]) != 0:
#                 size_cluster[i] = len(cluster_list[i])
#
#         sorted_size_cluster = sorted(size_cluster.items(), key=lambda x: x[1], reverse=True)
#         k = 0
#         while len(worker_combi) < worker_combi_num:
#             worker_combi.append(cluster_list[sorted_size_cluster[k][0]][0])
#             del cluster_list[sorted_size_cluster[k][0]][0]
#             if k + 1 == len(sorted_size_cluster):
#                 k = 0
#             else:
#                 k = k + 1
#         worker_combi_list.append(worker_combi)
#         i = i + 1
#     return worker_combi_list