import random
import collections
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import itertools
from statistics import mean
from clustering.same_size_kmeans import EqualGroupsKMeans
from clustering.balanced_kmeans_hungarian import balanced_kmeans
from clustering.kmedoids import BalancedKmedoids
from true_calc_cost import true_correct_correct_cost, true_correct_mis_cost, true_mis_dif_cost, true_mis_same_cost
from tqdm import tqdm


def read_file(breed_to_choice_dic):
    delete_list = list(range(15)) + list(range(16, 28)) + list(range(48, 102))  # + list(range(109, 117))
    all_df = pd.DataFrame()
    correct_answer_list = []
    for i in tqdm(range(40)):
        df = pd.DataFrame()
        for j in range(100):
            try:
                tmp_df = pd.read_csv('/Users/admin/Downloads/results/canis_q' + str(i + 1) + '_b' + str(j + 1) + '.csv')
                tmp_df.drop(columns=tmp_df.columns[delete_list], inplace=True)
                tmp_df.drop(columns='Reject', inplace=True)
                tmp_df.drop(columns='Approve', inplace=True)
                tmp_df = tmp_df.loc[:, ~tmp_df.columns.str.match('Answer\.img[0-9]{2}[2-7].*')]
                df = pd.concat([df, tmp_df])
            except Exception as e:
                break
        df.drop_duplicates(subset='WorkerId', keep='first', inplace=True)
        df = df.reset_index(drop=True)
        for label, items in df.iteritems():
            if label.startswith('Input'):
                correct_answer_list.append(breed_to_choice_dic[items[0][37:41]])
        df.drop(columns=df.columns[list(range(1, 21))], inplace=True)
        if i == 0:
            all_df = df
        else:
            all_df = pd.merge(all_df, df, on='WorkerId')
    all_df.drop(columns='WorkerId', inplace=True)
    pd.to_pickle(all_df, "all_df.pkl")
    pd.to_pickle(correct_answer_list, "correct_answer_list.pkl")
    return all_df, correct_answer_list


def make_confusion_matrix(correct_answer_list, worker_answer_list, choice_num):
    confusion_matrix_list = []
    accurate_list = []
    spammer_count = 0
    for i, worker_answers in enumerate(worker_answer_list):
        answer_count_matrix = np.zeros((choice_num, choice_num))
        correct_count = 0
        for worker_answer, correct_answer in zip(worker_answers, correct_answer_list):
            if worker_answer == correct_answer:
                correct_count = correct_count + 1
            answer_count_matrix[correct_answer, worker_answer] = answer_count_matrix[correct_answer, worker_answer] + 1
        confusion_matrix = answer_count_matrix / answer_count_matrix.sum(axis=1)[:, np.newaxis]
        confusion_matrix_list.append(confusion_matrix)
        accurate = float(correct_count) / 800.0
        if accurate < 0.40:
            spammer_count = spammer_count + 1
        accurate_list.append(accurate)
    # print("spammer_num: " + str(spammer_count))
    return confusion_matrix_list, accurate_list


def each_choice_prob(correct_answer_list, worker_answer_list, choice_num):
    choice_prob_list = []
    worker_num = len(worker_answer_list)
    for question_num, correct_answer in enumerate(correct_answer_list):
        each_choice_count = [0.0] * choice_num
        for worker_answer in worker_answer_list:
            each_choice_count[worker_answer[question_num]] = each_choice_count[worker_answer[question_num]] + 1.0
        each_choice_count = list(map(lambda x: x / float(worker_num), each_choice_count))
        choice_prob_list.append(each_choice_count)
    return choice_prob_list


def make_cost_matrix(correct_answer_list, worker_answer_list, choice_num):
    choice_prob_list = each_choice_prob(correct_answer_list, worker_answer_list, choice_num)
    worker_num = len(worker_answer_list)
    cost_matrix = np.zeros((worker_num, worker_num))
    for i in tqdm(range(worker_num)):
        for j in range(worker_num):
            if i <= j:
                cost_i_j = 0
                for q_num in range(len(correct_answer_list)):
                    if worker_answer_list[i][q_num] == correct_answer_list[q_num]:
                        if worker_answer_list[j][q_num] == correct_answer_list[q_num]:
                            cost = true_correct_correct_cost(choice_prob_list, q_num, correct_answer_list[q_num], choice_num)
                            cost_i_j = cost_i_j + cost
                        else:
                            cost = true_correct_mis_cost(choice_prob_list, q_num, correct_answer_list[q_num],
                                                    worker_answer_list[j][q_num], choice_num)
                            cost_i_j = cost_i_j + cost
                    elif worker_answer_list[j][q_num] == correct_answer_list[q_num]:
                        cost = true_correct_mis_cost(choice_prob_list, q_num, correct_answer_list[q_num],
                                                worker_answer_list[j][q_num], choice_num)
                        cost_i_j = cost_i_j + cost
                    elif worker_answer_list[i][q_num] == worker_answer_list[j][q_num]:
                        cost = true_mis_same_cost(choice_prob_list, q_num, correct_answer_list[q_num],
                                             worker_answer_list[i][q_num], choice_num)
                        cost_i_j = cost_i_j + cost
                    else:
                        cost = true_mis_dif_cost(choice_prob_list, q_num, correct_answer_list[q_num],
                                            worker_answer_list[i][q_num], worker_answer_list[j][q_num], choice_num)
                        cost_i_j = cost_i_j + cost
                cost_matrix[i, j] = cost_i_j / float(len(correct_answer_list))
                cost_matrix[j, i] = cost_i_j / float(len(correct_answer_list))
    return cost_matrix


def weight_answer_vectors(confusion_matrix_list, worker_answer_list):
    n_choices = confusion_matrix_list[0].shape[0]
    n_workers = len(confusion_matrix_list)
    weight_answer_vectors = np.empty((n_workers, n_choices * len(worker_answer_list[0])))
    for i, worker_answers in enumerate(worker_answer_list):
        for j, worker_answer in enumerate(worker_answers):
            for choice in range(n_choices):
                weight_answer_vectors[i, j * n_choices + choice] = confusion_matrix_list[i][worker_answer, choice]
    return weight_answer_vectors


def choice_teams(np_worker_vectors, cost_matrix, data_num, worker_combi_num, clustering):
    cluster_list = []
    for i in range(worker_combi_num):
        cluster_list.append([])
    data_ref_list = [num for num in range(100)]
    worker_combi_list = []
    selected_worker_list = []
    while len(worker_combi_list) < data_num / worker_combi_num:
        worker_combi = []
        has_empty = False
        for cluster in cluster_list:
            if len(cluster) == 0:
                has_empty = True
                break
        if has_empty:
            deleted_np_worker_vectors = np.delete(np_worker_vectors, selected_worker_list, 0)
            deleted_data_ref_list = np.delete(np.array(data_ref_list), selected_worker_list).tolist()

            if clustering == 'kmeans':
                pred = KMeans(n_clusters=worker_combi_num).fit_predict(deleted_np_worker_vectors)
            elif clustering == 'sskmeans':
                pred = EqualGroupsKMeans(n_clusters=worker_combi_num).fit_predict(deleted_np_worker_vectors)
            elif clustering == 'bkmeans':
                pred = balanced_kmeans(deleted_np_worker_vectors, worker_combi_num)
            elif clustering == 'dbscan':
                pred = DBSCAN(eps=15.0, min_samples=3).fit_predict(deleted_np_worker_vectors)
            elif clustering == 'bkmedoids':
                pred = BalancedKmedoids(n_cluster=worker_combi_num).fit_predict(cost_matrix)[0]

            cluster_list = []
            for i in range(worker_combi_num):
                cluster_list.append([])
            for i, cluster_num in enumerate(pred.tolist()):
                cluster_list[int(cluster_num)].append(deleted_data_ref_list[i])
            for i in range(len(cluster_list)):
                random.shuffle(cluster_list[i])
        for i in range(len(cluster_list)):
            worker_combi.append(cluster_list[i][0])
            del cluster_list[i][0]
        selected_worker_list = selected_worker_list + worker_combi
        worker_combi_list.append(worker_combi)
    return worker_combi_list


def pai_multi(confusiom_matrix_list, answers, prior_probability):
    c = collections.Counter(answers)
    majority_list = [x[0] for x in c.items() if x[1] == max(c.values())]
    pai_result = 0.0
    for majority_choice in majority_list:
        each_result = 1.0
        for j in range(len(confusiom_matrix_list)):
            each_result = each_result * confusiom_matrix_list[j][majority_choice, answers[j]]
        each_result = each_result * float(prior_probability[majority_choice])
        each_result = each_result / float(len(majority_list))
        pai_result = pai_result + each_result
    return pai_result


def expectation(confusiom_matrix_list, worker_combi_num, prior_probability):
    sum = 0.0
    all_choice_combination = itertools.product(list(range(len(prior_probability))), repeat=worker_combi_num)
    for choice_combination in all_choice_combination:
        a = pai_multi(confusiom_matrix_list, choice_combination, prior_probability)
        sum = sum + a
    return sum


def real_probability(worker_combi_list, worker_answer_list, correct_answer_list):
    probability_list = []
    for worker_combi in worker_combi_list:
        correct_num = 0.0
        for i, correct_answer in enumerate(correct_answer_list):
            answers = []
            for worker in worker_combi:
                answers.append(worker_answer_list[worker][i])
            c = collections.Counter(answers)
            majority_list = [x[0] for x in c.items() if x[1] == max(c.values())]
            for majority_choice in majority_list:
                if correct_answer == majority_choice:
                    correct_num = correct_num + 1.0 / float(len(majority_list))
        probability = correct_num / float(len(correct_answer_list))
        probability_list.append(probability)
    return mean(probability_list)


def weighted_real_probability(worker_combi_list, weight_answer_vectors, correct_answer_list, data_num, choice_num):
    probability_list = []
    weight_answer_vectors_reshape = weight_answer_vectors.reshape([data_num, -1, choice_num])
    for worker_combi in worker_combi_list:
        correct_num = 0.0
        for i, correct_answer in enumerate(correct_answer_list):
            answers = np.zeros((choice_num,))
            for worker in worker_combi:
                answers = answers * weight_answer_vectors_reshape[worker, i, :]
            majority = np.argmax(answers)
            if correct_answer == majority:
                correct_num = correct_num + 1.0
        probability = correct_num / float(len(correct_answer_list))
        probability_list.append(probability)
    return mean(probability_list)


def distance_ranking(np_worker_vectors, data_num):
    distance_dic = {}
    all_worker_pair = itertools.combinations(list(range(data_num)), 2)
    for worker_pair in all_worker_pair:
        worker_a = np_worker_vectors[worker_pair[0], :]
        worker_b = np_worker_vectors[worker_pair[1], :]
        worker_a_b = worker_a - worker_b
        distance = np.linalg.norm(worker_a_b)
        distance_dic[worker_pair] = distance
    sorted_distance_dic = sorted(distance_dic.items(), key=lambda x: x[1])
    return sorted_distance_dic
