import numpy as np
from sklearn.cluster import KMeans  # KMeans clustering
from sklearn.decomposition import PCA

from extract import load_cancer_for_neural
import matplotlib.pyplot as plt

for i in [30, 20, 15, 10, 5]:
    x_train, y_train, x_test, y_test, dims = load_cancer_for_neural()
    # print(x_train)

    kmeans = KMeans(n_clusters=2)

    pca = PCA(n_components=i)
    pca.fit(x_train)
    train_pca = pca.transform(x_train)

    pca = PCA(n_components=i)
    pca.fit(x_test)
    test_pca = pca.transform(x_test)

    kmeans.fit(x_train)

    X_clustered = kmeans.predict(x_test)

    pca = PCA(n_components=2)
    pca.fit(x_test)
    draw_components = pca.transform(x_test)
    ll = draw_components[X_clustered == 1]
    mm = draw_components[X_clustered == 0]
    print(ll.shape)

    plt.title(f"Zachowane cechy {i}")
    plt.scatter(ll[:, 0], ll[:,1])
    plt.scatter(mm[:, 0], mm[:,1])
    plt.show()