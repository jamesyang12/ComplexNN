"""
==================================================
Plot different SVM classifiers in the iris dataset
==================================================

Comparison of different linear SVM classifiers on a 2D projection of the iris
dataset. We only consider the first 2 features of this dataset:

- Sepal length
- Sepal width

This example shows how to plot the decision surface for four SVM classifiers
with different kernels.

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different decision boundaries. This can be a consequence of the following
differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.

- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes)
while the non-linear kernel models (polynomial or Gaussian RBF) have more
flexible non-linear decision boundaries with shapes that depend on the kind of
kernel and its parameters.

.. NOTE:: while plotting the decision function of classifiers for toy 2D
   datasets can help get an intuitive understanding of their respective
   expressive power, be aware that those intuitions don't always generalize to
   more realistic high-dimensional problems.

"""
#print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
#X = iris.data[:, :2]
y = iris.target

####
'''
    Data set:  Each species has 50 samples
    Three species, each species has four features

    Species: Setosa, Versicolour, and Virginica
    Features: Sepal Length, Sepal Width, Petal Length and Petal Width

'''
####

X_all = iris.data
print("Total --> ", X_all)
print("The data shape --> ", X_all.shape)
print("The target shape --> ", y.shape)
print(y)


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter

models = (svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='linear', C=C))

titles = ('SVC with linear kernel',
          'SVC with linear kernel',
          'SVC with linear kernel',
          'SVC with linear kernel',
          'SVC with linear kernel',
          'SVC with linear kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(3, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

#X0, X1 = X[:, 0], X[:, 1]
#xx, yy = make_meshgrid(X0, X1)
i=0
j=1

for clf, title, ax in zip(models, titles, sub.flatten()):
    if i==0:
      ax.set_xlabel('Sepal length')
    elif i==1:
      ax.set_xlabel('Sepal width')
    elif i==2:
      ax.set_xlabel('Petal length')
    elif i==3:
      ax.set_xlabel('Peta width')

    if j==0:
      ax.set_ylabel('Sepal length')
    elif j==1:
      ax.set_ylabel('Sepal width')
    elif j==2:
      ax.set_ylabel('Petal length')
    elif j==3:
      ax.set_ylabel('Petal width')

    X0, X1 = X_all[:, i], X_all[:, j]
    xx, yy = make_meshgrid(X0, X1)

    clf = svm.SVC(kernel='linear', C=C)
    XXX = np.column_stack((iris.data[:,i], iris.data[:,j]))
    clf.fit(XXX, y) 

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #ax.set_xlabel('Sepal length')
    #ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    
    if j>=3:
      i = i + 1
      j = i + 1
    else:
      j = j + 1

plt.show()
