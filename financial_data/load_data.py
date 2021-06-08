import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing


def drop_outliers(data, low=0.1, high=0.90):
    Q1 = data.quantile(low)
    Q3 = data.quantile(high)
    IQR = Q3 - Q1
    data = data[
        ~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data


def load_from_year(year: int, combine=False):


    if combine:
        data = pd.read_csv("2018_Financial_Data.csv")
        data = data.append(pd.read_csv("2017_Financial_Data.csv"))
        data = data.append(pd.read_csv("2016_Financial_Data.csv"))
        data = data.append(pd.read_csv("2015_Financial_Data.csv"))
        data = data.append(pd.read_csv("2014_Financial_Data.csv"))
    else:
        data = pd.read_csv(f"{year}_Financial_Data.csv")
    data = data.drop([f"{year + 1} PRICE VAR [%]"], axis=1)
    data = data.fillna(data.mean())
    data = drop_outliers(data)
    return data


def prepare_data_from(data, yearly=False, one_hot= True):
    output_data = data["Class"].values
    input_data = data.drop(["Class", "ShortName", "Sector"], axis=1).values

    print(input_data.shape)

    if yearly:
        x_train = input_data
        y_train = output_data
        x_test = input_data
        y_test = output_data
    else:
        y_train = output_data[:int(output_data.shape[0] / 2)]
        y_test = output_data[int(output_data.shape[0] / 2):]

        x_train = input_data[:int(input_data.shape[0] / 2)]
        x_test = input_data[int(input_data.shape[0] / 2):]

    print(x_train, x_test)
    if one_hot:
        y_train = tf.one_hot(y_train, 2)
        y_test = tf.one_hot(y_test, 2)

    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)
    return x_train, y_train, x_test, y_test
