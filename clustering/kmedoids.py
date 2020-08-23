import numpy as np
import pandas as pd
from munkres import Munkres
import random


class Kmedoids(object):

    def __init__(self, n_cluster, max_iter=500):
        self.n_cluster = n_cluster
        self.max_iter = max_iter

    def fit_predict(self, distance_matrix):
        n_samples = distance_matrix.shape[0]

        # STEP1

        # 1-2評価値vの計算
        v = {}
        for j in range(n_samples):
            v_j = 0.0
            for i in range(n_samples):
                sum_distance_i = 0.0
                for l in range(n_samples):
                    sum_distance_i = sum_distance_i + distance_matrix[i, l]
                v_j = v_j + distance_matrix[i, j] / sum_distance_i
            v[j] = v_j
        v = sorted(v.items(), key=lambda x: x[1])

        # 1-3初期medoidの決定
        medoids = []
        for i in range(self.n_cluster):
            medoids.append(v[i][0])

        # 最大max_iter回のmedoid更新
        ith_distance = 0.0
        for i_update in range(self.max_iter):

            # 1-4初期クラスタ(最初の割り当て)
            assigned_cluster_list = []
            cluster = []
            distance_dic = {}
            for i in range(self.n_cluster):
                cluster.append([])
            for i in range(n_samples):
                medoid_distance = []
                for medoid in medoids:
                    medoid_distance.append(distance_matrix[i, medoid])
                assign_medoid = medoid_distance.index(min(medoid_distance))
                assigned_cluster_list.append(assign_medoid)
                cluster[assign_medoid].append(i)
                distance_dic[i] = medoid_distance[assign_medoid]

            # クラスタ0のところに強制割り当て
            distance_dic = sorted(distance_dic.items(), key=lambda x: x[1], reverse=True)
            for i in range(self.n_cluster):
                if len(cluster[i]) == 0:
                    for j in range(n_samples):
                        data = distance_dic[j][0]
                        if len(cluster[assigned_cluster_list[data]]) > 2:
                            break
                    cluster[i].append(data)
                    cluster[assigned_cluster_list[data]].remove(data)
                    assigned_cluster_list[data] = i

            # 1-5それぞれのmedoidまでの距離の総和
            sum_distance_to_medoid = 0.0
            for i in range(n_samples):
                assign_medoid = assigned_cluster_list[i]
                sum_distance_to_medoid = sum_distance_to_medoid + distance_matrix[i, medoids[assign_medoid]]
            if ith_distance == sum_distance_to_medoid:
                print(str(i) + "回目でbreak")
                break
            ith_distance = sum_distance_to_medoid

            # STEP2
            # メドイドの更新
            medoids = []
            for i in range(self.n_cluster):
                sum_distance_exself_list = []
                for j in cluster[i]:
                    sum_distance_exself = 0.0
                    for k in cluster[i]:
                        if k != j:
                            sum_distance_exself = sum_distance_exself + distance_matrix[j, k]
                    sum_distance_exself_list.append(sum_distance_exself)
                new_medoid_index = sum_distance_exself_list.index(min(sum_distance_exself_list))
                new_medoid = cluster[i][new_medoid_index]
                medoids.append(new_medoid)

        return assigned_cluster_list, cluster


class BalancedKmedoids(object):

    def __init__(self, n_cluster, max_iter=300):
        self.n_cluster = n_cluster
        self.max_iter = max_iter

    def fit_predict(self, distance_matrix):
        n_samples = distance_matrix.shape[0]

        # STEP1

        # 1-2評価値vの計算
        v = {}
        for j in range(n_samples):
            v_j = 0.0
            for i in range(n_samples):
                sum_distance_i = 0.0
                for l in range(n_samples):
                    sum_distance_i = sum_distance_i + distance_matrix[i, l]
                v_j = v_j + distance_matrix[i, j] / sum_distance_i
            v[j] = v_j
        v = sorted(v.items(), key=lambda x: x[1])

        # 1-3初期medoidの決定
        medoids = []
        for i in range(self.n_cluster):
            medoids.append(v[i][0])

        # 最大max_iter回のmedoid更新
        ith_distance = 0.0
        for i_update in range(self.max_iter):

            slot_distance_matrix = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for slot in range(n_samples):
                    medoid_num = slot % self.n_cluster
                    slot_distance_matrix[i, slot] = distance_matrix[i, medoids[medoid_num]]

            m = Munkres()
            pred = m.compute(slot_distance_matrix)

            assigned_cluster_list = []
            cluster = []
            for i in range(self.n_cluster):
                cluster.append([])
            for i in range(n_samples):
                assign_medoid = pred[i][1] % self.n_cluster
                assigned_cluster_list.append(assign_medoid)
                cluster[assign_medoid].append(i)

            # 1-5それぞれのmedoidまでの距離の総和
            sum_distance_to_medoid = 0.0
            for i in range(n_samples):
                assign_medoid = assigned_cluster_list[i]
                sum_distance_to_medoid = sum_distance_to_medoid + distance_matrix[i, medoids[assign_medoid]]
            if ith_distance == sum_distance_to_medoid:
                print(str(i) + "回目でbreak")
                break
            ith_distance = sum_distance_to_medoid

            # STEP2
            # メドイドの更新
            medoids = []
            for i in range(self.n_cluster):
                sum_distance_exself_list = []
                for j in cluster[i]:
                    sum_distance_exself = 0.0
                    for k in cluster[i]:
                        if k != j:
                            sum_distance_exself = sum_distance_exself + distance_matrix[j, k]
                    sum_distance_exself_list.append(sum_distance_exself)
                new_medoid_index = sum_distance_exself_list.index(min(sum_distance_exself_list))
                new_medoid = cluster[i][new_medoid_index]
                medoids.append(new_medoid)

        return assigned_cluster_list, cluster


class RandomBalancedKmedoids(object):

    def __init__(self, n_cluster, max_iter=300):
        self.n_cluster = n_cluster
        self.max_iter = max_iter

    def fit_predict(self, distance_matrix):
        n_samples = distance_matrix.shape[0]

        # STEP1

        # 1-3初期medoidの決定
        medoids = random.sample(list(range(n_samples)), self.n_cluster)

        # 最大max_iter回のmedoid更新
        ith_assigned_cluster = []
        for i_update in range(self.max_iter):

            # 1-4初期クラスタ(最初の割り当て)
            assigned_cluster_list = []
            cluster = []
            distance_dic = {}
            for i in range(self.n_cluster):
                cluster.append([])
            for i in range(n_samples):
                medoid_distance = []
                for medoid in medoids:
                    medoid_distance.append(distance_matrix[i, medoid])
                assign_medoid = medoid_distance.index(min(medoid_distance))
                assigned_cluster_list.append(assign_medoid)
                cluster[assign_medoid].append(i)
                distance_dic[i] = medoid_distance[assign_medoid]

            # クラスタ0のところに強制割り当て
            distance_dic = sorted(distance_dic.items(), key=lambda x: x[1], reverse=True)
            for i in range(self.n_cluster):
                if len(cluster[i]) == 0:
                    for j in range(n_samples):
                        data = distance_dic[j][0]
                        if len(cluster[assigned_cluster_list[data]]) > 2:
                            break
                    cluster[i].append(data)
                    cluster[assigned_cluster_list[data]].remove(data)
                    assigned_cluster_list[data] = i

            # 更新続けるかをチェック
            if ith_assigned_cluster == assigned_cluster_list:
                break
            ith_assigned_cluster = assigned_cluster_list

            # STEP2
            # メドイドの更新
            medoids = []
            for i in range(self.n_cluster):
                sum_distance_exself_list = []
                for j in cluster[i]:
                    sum_distance_exself = 0.0
                    for k in cluster[i]:
                        if k != j:
                            sum_distance_exself = sum_distance_exself + distance_matrix[j, k]
                    sum_distance_exself_list.append(sum_distance_exself)
                new_medoid_index = sum_distance_exself_list.index(min(sum_distance_exself_list))
                new_medoid = cluster[i][new_medoid_index]
                medoids.append(new_medoid)

        return assigned_cluster_list, cluster
