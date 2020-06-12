import numpy as np
from munkres import Munkres


def balanced_kmeans(X, n_clusters, max_iter=300):
    n_samples, n_features = X.shape

    # ランダムに重心の初期値を初期化
    centroids = X[np.random.choice(n_samples, n_clusters)]

    # 前の重心と比較するために、仮に新しい重心を入れておく配列を用意
    new_centroids = np.zeros((n_clusters, n_features))

    # 各データ所属クラスタ情報を保存する配列を用意
    cluster = np.zeros(n_samples)

    # ループ上限回数まで繰り返し
    for epoch in range(max_iter):
        slot_distance_matrix = np.zeros((n_samples, n_samples))
        # 入力データ全てに対して繰り返し
        for i in range(n_samples):
            # データから各重心までの距離を計算（ルートを取らなくても大小関係は変わらないので省略）
            distances = np.sum((centroids - X[i]) ** 2, axis=1)
            for slot in range(n_samples):
                cluster_num = slot % n_clusters
                slot_distance_matrix[i, slot] = distances[cluster_num]

        m = Munkres()
        pred = m.compute(slot_distance_matrix)

        for i in range(n_samples):
            cluster[i] = pred[i][1] % n_clusters
        # データの所属クラスタを距離の一番近い重心を持つものに更新
        # cluster[i] = np.argsort(distances)[0]

        # すべてのクラスタに対して重心を再計算
        for j in range(n_clusters):
            new_centroids[j] = X[cluster == j].mean(axis=0)

        # もしも重心が変わっていなかったら終了
        if np.sum(new_centroids == centroids) == n_clusters:
            print("break")
            break
        centroids = new_centroids
    return cluster
