print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from complex_nn import neuralNetwork

iris = datasets.load_iris()
X = iris.data
y = iris.target

n_sample = len(X)

# np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.float)

train_per = .75
X_train = X[:int(train_per * n_sample)]
y_train = y[:int(train_per * n_sample)]
X_test = X[int(train_per * n_sample):]
y_test = y[int(train_per * n_sample):]

# fit the model
kernels = ["linear", "rbf", "poly"]
for k in kernels:
    print(k)
    clf = SVC(kernel=k)
    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))


# create instance of neural network
number_of_inputs = 4
categories = 3
periods = 2

n = neuralNetwork(number_of_inputs, categories, periods)
n.status()

def calcScore():
    scorecard = []
    for i in range(X_test.shape[0]):
        y_pred = n.query(list(X_test[i,:]))
        if y_pred == y_test[i]:
            scorecard.append(1)
        else:
            scorecard.append(0)

    return np.sum(scorecard)/len(scorecard)

X_train = X_train * 0.01
X_test = X_test * 0.01

epochs = 100
score = 0.
num_iter = 0
# for e in range(epochs):
while score < .9 and num_iter < 1000:
    # go through all records in the training data set
    # data_list = data.tolist()
    # species = data_list[0]
    # lengths = data_list[1:]
    for i in range(X_train.shape[0]):
        n.train(list(X_train[i,:]), y_train[i])
    pass

    # n.status()
    print(num_iter)
    if (num_iter+1) % 10 == 0:
        score = calcScore()
        print(score)
        n.status()
    num_iter += 1

