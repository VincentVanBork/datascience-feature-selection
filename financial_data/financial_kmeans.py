from itertools import combinations

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from financial_data.load_data import load_from_year, prepare_data_from

for num_components in [180, 120, 60, 30]:
    data = load_from_year(2018)

    x_train, y_train, x_test, y_test = prepare_data_from(data, yearly=True)

    pca = PCA(n_components=num_components)
    pca.fit(x_train)
    train_pca = pca.transform(x_train)

    pca = PCA(n_components=num_components)
    pca.fit(x_test)
    test_pca = pca.transform(x_test)

    kmeans = KMeans(n_clusters=2)

    kmeans.fit(x_train)

    X_clustered = kmeans.predict(x_test)

    num_of_draw = 4
    pca = PCA(n_components=num_of_draw)
    pca.fit(x_test)
    draw_components = pca.transform(x_test)
    ll = draw_components[X_clustered == 1]
    mm = draw_components[X_clustered == 0]

    # plt.title(f"Zachowane cechy {num_components}")

    fig, ax = plt.subplots(nrows=3, ncols=2)
    fig.suptitle(f'Zachowane cechy: {num_components}', fontsize=16)
    fig.tight_layout()
    for x,ax in zip(combinations(range(num_of_draw), 2),ax.reshape(-1)):
        i, j = x
        ax.scatter(ll[:, i], ll[:, j], s=1)
        ax.scatter(mm[:, i], mm[:, j], s=1)
        print(i, j)
        # ax[i, j].scatter(ll[:, i], ll[:, j], s=1)
        # ax[i, j]
        #
        # ax[i, j].get_xaxis().set_visible(False)
        # ax[i, j].get_yaxis().set_visible(False)
        # # ax[i,j].scatter(mm[:, i], mm[:, j])
    plt.show()
