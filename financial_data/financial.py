import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

# data = pd.read_csv("2017_Financial_Data.csv")
# data = data.append(pd.read_csv("2018_Financial_Data.csv"))
# data = data.append(pd.read_csv("2016_Financial_Data.csv"))
# data = data.append(pd.read_csv("2015_Financial_Data.csv"))
# data = data.append(pd.read_csv("2014_Financial_Data.csv"))
from sklearn.decomposition import PCA

from financial_data.load_data import load_from_year, prepare_data_from
for number_components in [180, 120, 60, 30, 20]:
    pca = PCA(n_components=number_components)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(number_components, activation='relu'),
        # tf.keras.layers.Dense(int(number_components * 2), activation='relu'),
        # tf.keras.layers.Dense(int(number_components * 4), activation='relu'),
        # tf.keras.layers.Dense(int(number_components * 8), activation='relu'),
        # tf.keras.layers.Dense(int(number_components * 16), activation='relu'),
        # tf.keras.layers.Dense(int(number_components/2), activation='relu'),
        tf.keras.layers.Dense(int(number_components/4), activation='relu'),
        tf.keras.layers.Dense(int(number_components*2), activation='relu'),
        tf.keras.layers.Dense(int(number_components /6), activation='relu'),
        # tf.keras.layers.Dense(int(number_components / 8), activation='relu'),
        # tf.keras.layers.Dense(int(number_components / 16), activation='relu'),
        # tf.keras.layers.Dense(int(number_components / 32), activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    epochs = 40

    fig, ax = plt.subplots(ncols=2)
    year = 2018

    print("YEAR", year)
    data = load_from_year(year, combine=True)

    x_train, y_train, x_test, y_test = prepare_data_from(data, yearly=True)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    print("INPUT FEATURES", x_train.shape[1])
    history = model.fit(x_train, y_train, epochs=epochs, verbose=0)
    results_test = model.evaluate(x_test, y_test)
    print("RESULTS FROM EVALUATION", results_test)
    ax[0].plot(history.history['loss'])
    ax[1].plot(history.history['accuracy'])

    ax[0].plot(epochs, results_test[0], 'ro')
    ax[1].plot(epochs, results_test[1], 'ro')

    ax[1].set_title("accuracy")
    ax[0].set_title("loss")

    ax[0].set_xlabel("epoch")
    ax[1].set_xlabel("epoch")
    plt.show()

    # fig_variance, ax_variance = plt.subplots()
    # ax_variance.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    # plt.show()


    print("STARTING PREDICTION")
    data = load_from_year(2017)
    x_train, y_train, x_test, y_test = prepare_data_from(data, yearly=True)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    print("INPUT FEATURES", x_train.shape[1])
    results_test = model.evaluate(x_test, y_test)
    print("RESULTS FINAL", results_test)
    # plt.legend(['train', 'test'], loc='upper left')
    plt.show()

