import numpy as np
from munkres import Munkres


def k_init(X, n_clusters):
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    first_center_id = np.random.randint(n_samples)
    centers[0] = X[first_center_id]

    # 最初のクラスタ点とそれ以外のデータ点との距離の2乗を計算し、それぞれをその総和で割る
    prob_chosen_center = ((X - centers[0]) ** 2).sum(axis=1) / ((X - centers[0]) ** 2).sum()
    second_center_id = np.random.choice(np.array(range(n_samples)), size=1, replace=False, p=prob_chosen_center)
    centers[1] = X[second_center_id]

    # ランダムに最初のクラスタ点を決定
    tmp = np.random.choice(np.array(range(X.shape[0])))
    first_cluster = X[tmp]
    first_cluster = first_cluster[np.newaxis, :]

    # 最初のクラスタ点とそれ以外のデータ点との距離の2乗を計算し、それぞれをその総和で割る
    p = ((X - first_cluster) ** 2).sum(axis=1) / ((X - first_cluster) ** 2).sum()

    r = np.random.choice(np.array(range(X.shape[0])), size=1, replace=False, p=p)

    first_cluster = np.r_[first_cluster, X[r]]

    # 分割するクラスター数が3個以上の場合
    if n_clusters >= 3:
        # 指定の数のクラスタ点を指定できるまで繰り返し
        while first_cluster.shape[0] < n_clusters:
            # 各クラスター点と各データポイントとの距離の2乗を算出
            dist_f = ((X[:, :, np.newaxis] - first_cluster.T[np.newaxis, :, :]) ** 2).sum(axis=1)
            # 最も距離の近いクラスター点はどれか導出
            f_argmin = dist_f.argmin(axis=1)
            # 最も距離の近いクラスター点と各データポイントとの距離の2乗を導出
            for i in range(dist_f.shape[1]):
                dist_f.T[i][f_argmin != i] = 0

            # 新しいクラスタ点を確率的に導出
            pp = dist_f.sum(axis=1) / dist_f.sum()
            rr = np.random.choice(np.array(range(X.shape[0])), size=1, replace=False, p=pp)
            # 新しいクラスター点を初期値として加える
            first_cluster = np.r_[first_cluster, X[rr]]

    return centers


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
