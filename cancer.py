import tensorflow as tf
from matplotlib import pyplot as plt

from extract import load_cancer_for_neural
from sklearn.decomposition import PCA

for i in [30, 20, 15, 10, 5]:
    pca = PCA(n_components=i)

    x_train, y_train, x_test, y_test, dims = load_cancer_for_neural()

    pca.fit(x_train)

    # print(x_test.shape)
    # print(y_test.shape)

    train_pca = pca.transform(x_train)
    test_pca = pca.transform(x_test)

    # print(x_train.shape, train_pca.shape)
    # print(x_test.shape, test_pca.shape)

    # print("________________________")
    print("NUMBER OF COMPONENTS", pca.n_components_)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(pca.n_components_, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        # tf.keras.layers.Dense(8, activation='relu'),
        # tf.keras.layers.Dense(8, activation='relu'),
        # tf.keras.layers.Dense(8, activation='relu'),
        # tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    epochs = 15
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if not pca:
        history = model.fit(x_train, y_train, epochs=epochs)
    else:
        history = model.fit(train_pca, y_train, epochs=epochs, verbose=0)

    # print("EVALUATING")
    if not pca:
        results_test = model.evaluate(x_test, y_test)
    else:
        results_test = model.evaluate(test_pca, y_test, verbose=0)

    print("TEST RESULTS (loss, acc)", results_test)
    # print("VARIANCE", pca.explained_variance_)
    fig_variance, ax_variance = plt.subplots()
    ax_variance.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.show()

    # fig, ax = plt.subplots(ncols=2)
    #
    # ax[0].plot(history.history['loss'])
    # ax[1].plot(history.history['accuracy'])
    #
    # ax[0].plot(epochs, results_test[0], 'ro')
    # ax[1].plot(epochs, results_test[1], 'ro')
    #
    # ax[1].set_title("accuracy")
    # ax[0].set_title("loss")
    #
    # ax[0].set_xlabel("epoch")
    # ax[1].set_xlabel("epoch")
    #
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
