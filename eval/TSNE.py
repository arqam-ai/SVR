import os
import pickle
import numpy as np
from collections import Counter
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib
import matplotlib.pyplot as plt

def load_prediction(filepath):
    """load prediction index
    Params:
    ------------------
    filepath : string
        path to prediction index file

    Returns:
    ------------------
    pred_list : list
        list of cluster idx (prediction)
    """
    f = open(filepath)
    pred_list = []
    counter = 0
    for idx, pred in enumerate(f):
        if idx % 5 == 0:
            pred_list.append(int(pred[:-1]))

    return pred_list


with open("../voxel/test_data_filename.pkl", "rb") as f:
    test_filename = pickle.load(f)

pred_list = load_prediction("../what3d_clusters/predictions.txt")
pred_list = np.asarray(pred_list)


# class_list = {}
# for file in test_filename:
#     class_type = file.split("/")[0]
#     if class_type not in class_list:
#         class_list[class_type] = 1
#     else:
#         class_list[class_type] += 1
#
# overall_class = [k for k in class_list.keys()]
# overall_class.sort()
# class_index = []
# for file in test_filename:
#     class_type = file.split("/")[0]
#     idx = overall_class.index(class_type)
#     class_index.append(idx)
# class_index = np.asarray(class_index)


###################### Plot Clustering ######################
distance_matrix = np.load("distance_matrix/Cluster_all_distance_matrix.npy")  #distance_matrix/Cluster_all_distance_matrix.npy Pred_all_distance_matrix.npy
model = TSNE(perplexity=30, n_jobs=8, metric="precomputed")
embeddings = model.fit_transform(distance_matrix)
# np.save("Pred_embedding.npy", embeddings)

# embeddings = np.load("clustering_embedding.npy") # "Pred_embedding.npy" "clustering_embedding.npy"

# LINE_STYLES = ['.', 'o', '+', 'dotted']
NUM_COLORS = 500
# # cm = plt.get_cmap('gist_rainbow')
# cmap = plt.get_cmap('jet')
# colors = cmap(np.linspace(0, 1.0, 500))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in range(NUM_COLORS):
#     data_idx = np.where(pred_list[:]==i)[0]
#     if i == 1:
#         c = 2
#     if len(data_idx) == 0:
#         continue
#     cx, cy = embeddings[data_idx, 0], embeddings[data_idx, 1]
#     # ax.plot(cx, cy, marker='.', markersize=2, linestyle='None', color=float(i) / 500.)
#     # ax.scatter(cx, cy, c=float(i) / 500.)
#     # lines = ax.plot(cx, cy, marker='.', markersize=2, linestyle='None', colors=float(i)/500.)
#     # x = 1
#     # lines[0].set_color(rgba_color)
#     # lines[0].set_color(cm(i//3*3.0/NUM_COLORS))
#     # lines[0].set_color(cm(i / NUM_COLORS))

cluster = np.arange(0,NUM_COLORS)
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 1.0, 500))
fig = plt.figure()
ax = fig.add_subplot(111)
for cl, color in zip(cluster, colors):
    data_idx = np.where(pred_list[:]==cl)[0]
    if len(data_idx) == 0:
        continue
    cx, cy = embeddings[data_idx, 0], embeddings[data_idx, 1]
    plt.plot(cx, cy, '.', linestyle='None', color=color, markersize=2)

plt.xticks([], [])
plt.yticks([], [])
plt.xlim([-70, 70])
plt.ylim([-70, 70])
fig.savefig('Clustering_TSNE.png')
plt.show()

###################### Plot Clustering ######################
