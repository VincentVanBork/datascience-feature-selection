from itertools import combinations

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans  # KMeans clustering
from sklearn.decomposition import PCA
import pandas as pd


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

num_components = 4
for num_components in [38, 20, 10, 5, 2]:
    df = pd.read_csv('nndb_flat.csv')
    df = df.drop(
        ["ID", "FoodGroup", "ShortDescrip", "Descrip", "CommonName", "MfgName",
         "ScientificName"], axis=1)
    df = maximum_absolute_scaling(df)

    pca = PCA(n_components=num_components)
    input_values = pca.fit_transform(df.values)

    num_clasters = 25
    kmeans = KMeans(n_clusters=num_clasters)

    X_clustered = kmeans.fit_predict(input_values)

    num_of_draw = 4
    pca = PCA(n_components=num_of_draw)
    pca.fit(df.values)
    draw_components = pca.transform(df.values)
    ll = draw_components[X_clustered == 1]
    mm = draw_components[X_clustered == 0]
    print(ll.shape)

    fig, ax = plt.subplots(nrows=3, ncols=2)
    fig.suptitle(f'Zachowane cechy: {num_components}', fontsize=16)
    fig.tight_layout()
    for x, ax in zip(combinations(range(num_of_draw), 2), ax.reshape(-1)):
        i, j = x
        for cluster in range(num_clasters):
            ax.scatter(draw_components[X_clustered == cluster][:, i],
                       draw_components[X_clustered == cluster][:, j], s=1)
        print(i, j)
    plt.show()
