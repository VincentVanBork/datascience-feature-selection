import numpy as np
from matplotlib import pyplot as plt
from minisom import MiniSom


from financial_data.load_data import load_from_year, prepare_data_from
data = load_from_year(2018)

x_train, y_train, x_test, y_test = prepare_data_from(data, yearly=True, one_hot=False)
net_shape = (221*2, 221*2)

# print(iris_numpy)

som = MiniSom(net_shape[0], net_shape[1], x_train.shape[1], net_shape[1]/2.2, learning_rate=0.5, random_seed=6578)
som.train_batch(x_train, 10, verbose=True)

markers = ['o', 's']
colors = ['C0', 'C1']

plt.figure(figsize=(12,12))

plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
plt.colorbar()

print(y_train)
for cnt, xx in enumerate(x_train):
    print("doing sth", cnt)
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0] + .5, w[1] + .5, markers[y_train[cnt] - 1], markerfacecolor='None',
             markeredgecolor=colors[y_train[cnt] - 1], markersize=12, markeredgewidth=5)
plt.legend()
plt.show()