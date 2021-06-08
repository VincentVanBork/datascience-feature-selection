import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
import tensorflow as tf

data = genfromtxt('dota2Train.csv', delimiter=',')
output_data_train = data[:, 0]
input_data_train = data[:, 1:]

y_train = output_data_train
x_train = input_data_train

data_test = genfromtxt("dota2Test.csv", delimiter=',')
output_data_test = data_test[:, 0]
input_data_test = data_test[:, 1:]

y_test = output_data_test
x_test = input_data_test

y_train = np.where(y_train == -1, 0, y_train)
y_test = np.where(y_test == -1, 0, y_test)

# x_train = np.where(x_train == -1, 2, x_train)
# x_test = np.where(x_test == -1, 2, x_test)

# y_train = y_train / y_train.max(axis=0)
x_train = x_train / x_train.max(axis=0)
x_train = np.nan_to_num(x_train)

# y_test = y_test / y_test.max(axis=0)
x_test = x_test / x_test.max(axis=0)
x_test = np.nan_to_num(x_test)

y_train = tf.one_hot(y_train, 2)
y_test = tf.one_hot(y_test, 2)

print(x_train.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(230, input_shape=(x_train.shape[1],) ,activation='relu'),
    tf.keras.layers.Dense(46, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Ftrl(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
epochs = 40
history = model.fit(x_train, y_train, epochs=epochs)
results_test = model.evaluate(x_test, y_test)
fig, ax = plt.subplots(ncols=2)

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
