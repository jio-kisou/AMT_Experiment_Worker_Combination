import numpy as np
from sklearn.preprocessing import OneHotEncoder
from amt_func import read_file, make_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
import pandas as pd


data_num = 30
breed_to_choice_dic = {'Coyo': 0, 'Dhol': 1, 'Husk': 2, 'Alas': 3, 'Samo': 4, 'Germ': 5, 'wolf': 6}
choice_num = len(breed_to_choice_dic)

# データ読み込み
# all_df, correct_answer_list = read_file(breed_to_choice_dic)
all_df = pd.read_pickle("all_df.pkl")
correct_answer_list = pd.read_pickle("correct_answer_list.pkl")

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

# PCA
pca = PCA(n_components=2)
pca.fit(one_hot_data)
pca_trans = pca.transform(one_hot_data)

# 可視化
fig = plt.figure(figsize=(8,6))
plt.scatter(pca_trans[:, 0], pca_trans[:, 1])
plt.savefig("PCA_Image/pca_trans_" + str(data_num) + ".png")
# plt.show()

# KernelPCA
kernel_pca = KernelPCA(n_components=2)
kernel_pca.fit(one_hot_data)
kernel_pca_trans = kernel_pca.transform(one_hot_data)

# 可視化
fig = plt.figure(figsize=(8,6))
plt.scatter(kernel_pca_trans[:, 0], kernel_pca_trans[:, 1])
plt.savefig("KernelPCA_Image/kernel_pca_trans_" + str(data_num) + ".png")
# plt.show()

# # Isomap
# isomap = Isomap(n_neighbors=4, n_components=2)
# isomap.fit(one_hot_data)
# isomap_trans = isomap.transform(one_hot_data)
#
# # 可視化
# fig = plt.figure(figsize=(8,6))
# plt.scatter(isomap_trans[:, 0], isomap_trans[:, 1])
# plt.savefig("Isomap_Image/isomap_trans_" + str(data_num) + ".png")
# # plt.show()

# LocallyLinearEmbedding
locally_linear_embedding = LocallyLinearEmbedding(n_neighbors=5, n_components=2)
locally_linear_embedding.fit(one_hot_data)
locally_linear_embedding_trans = locally_linear_embedding.transform(one_hot_data)

# 可視化
fig = plt.figure(figsize=(8,6))
plt.scatter(locally_linear_embedding_trans[:, 0], locally_linear_embedding_trans[:, 1])
plt.savefig("LocallyLinearEmbedding_Image/locally_linear_embedding_trans_" + str(data_num) + ".png")
# plt.show()

# tSNE
tSNE = TSNE(n_components=2, perplexity=30.0)
tSNE_trans = tSNE.fit_transform(one_hot_data)

# 可視化
fig = plt.figure(figsize=(8,6))
plt.scatter(tSNE_trans[:, 0], tSNE_trans[:, 1])
plt.savefig("tSNE_Image/tSNE_trans_" + str(data_num) + ".png")
# plt.show()
