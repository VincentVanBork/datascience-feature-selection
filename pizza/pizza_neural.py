import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from minisom import MiniSom
from sklearn.decomposition import PCA


def maximum_absolute_scaling(adf):
    # copy the dataframe
    df_scaled = adf.copy()
    # apply maximum absolute scaling
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column] / df_scaled[column].abs().max()
    return df_scaled

df = pd.read_csv('nndb_flat.csv')
df = df.drop(
    ["ID", "FoodGroup", "ShortDescrip", "Descrip", "CommonName", "MfgName",
     "ScientificName"], axis=1)
df = maximum_absolute_scaling(df)

def drop_outliers(data, low=0.01, high=0.9):
    Q1 = data.quantile(low)
    Q3 = data.quantile(high)
    IQR = Q3 - Q1
    data = data[
        ~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data

df = drop_outliers(df)
net_shape = (1, 4)

# ldf = pd.read_csv('nndb_flat.csv')["FoodGroup"].str.get_dummies()
# print(ldf.info())
# print(ldf.head())

num_components = 38
pca = PCA(n_components=num_components)
x_train = pca.fit_transform(df.values)

# fig_variance, ax_variance = plt.subplots()
# ax_variance.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
# plt.show()

som = MiniSom(net_shape[0], net_shape[1], x_train.shape[1], sigma=0.5, learning_rate=0.5, random_seed=6578)
som.train_batch(x_train, 10, verbose=True)

markers = ['o', 's']
colors = ['C0', 'C1']

# plt.figure(figsize=(12,12))
#
# plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
# plt.colorbar()


# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in x_train]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, net_shape)

fig_clusters, ax_clusters = plt.subplots()

for c in np.unique(cluster_index):
    ax_clusters.scatter(x_train[cluster_index == c, 0],
                x_train[cluster_index == c,1], label='cluster='+str(c), alpha=.7, s=1)

# plotting centroids
for centroid in som.get_weights():
    ax_clusters.scatter(centroid[:, 0], centroid[:,1], marker='x',
                s=2, linewidths=8, color='k', label='centroid')
plt.legend()



plt.show()