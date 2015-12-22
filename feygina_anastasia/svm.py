from sklearn import svm
from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt
from data_preparation import *


def try_svm_method(train, test, n):
    # features that will be used for training:
    x = train.iloc[:, 0:20].values
    # target classes
    y = train.values[:, -1]
    # SVM regularization parameter
    c = 1
    # train model
    # svc = svm.SVC(kernel='linear', c=1)
    # poly_svc = svm.SVC(kernel='poly', degree=3, c=1)
    # lin_svc = svm.LinearSVC(C=c).fit(x, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=c)
    # cross_val_score
    scores = cross_validation.cross_val_score(rbf_svc, x, y, cv=n)
    # results of prediction
    print("Score:", np.mean(scores))
    # train model again
    rbf_svc.fit(x, y)
    # test features
    features = test.iloc[:, 0:20].values
    # results of prediction
    print("Results of prediction: ", rbf_svc.predict(features))

# h = .02
# svc = svm.SVC(kernel='linear', C=C).fit(x, y)
# rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(x, y)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x, y)
# lin_svc = svm.LinearSVC(C=C).fit(x, y)

# # create a mesh to plot in
# x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
# y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
#
# # title for the plots
# titles = ['SVC with linear kernel',
#           'LinearSVC (linear kernel)',
#           'SVC with RBF kernel',
#           'SVC with polynomial (degree 3) kernel']
#
# for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, m_max]x[y_min, y_max].
#     plt.subplot(2, 2, i + 1)
#     plt.subplots_adjust(wspace=0.4, hspace=0.4)
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#
#     # Plot also the training points
#     plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
#     plt.xlabel('Sepal length')
#     plt.ylabel('Sepal width')
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.xticks(())
#     plt.yticks(())
#     plt.title(titles[i])
#
# plt.show()
