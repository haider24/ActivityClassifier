import glob
import os
import pandas as pd
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def readDirectory(path):
    allFiles = glob.glob(os.path.join(path, "*.txt"))  # advisable to use os.path.join as this makes concatenation OS independent
    dataFrame = (pd.read_csv(f, header=None, delimiter=r"\s+", names=["X", "Y", "Z"]) for f in allFiles)
    data = pd.concat(dataFrame, ignore_index=True)
    return data

def getData():
    sitting = readDirectory(r'Data\Sitting')
    walking = readDirectory(r'Data\Walking')
    climbingStairs = readDirectory(r'Data\Climbing Stairs')
    lying = readDirectory(r'Data\Lying')
    sitting['Activity'] = 'Sitting'
    walking['Activity'] = 'Walking'
    climbingStairs['Activity'] = 'Climbing Stairs'
    lying['Activity'] = 'Lying'
    frames = [sitting, walking, climbingStairs, lying]
    dataSet = pd.concat(frames)
    dataSet = dataSet.sample(frac=1).reset_index(drop=True)
    return dataSet


def testModel(model,dataSet,X,Y):
    model.fit(dataSet[X], dataSet[Y])

    predictions = model.predict(dataSet[X])

    accuracy = metrics.accuracy_score(predictions, dataSet[Y])
    print("Training Accuracy : %s" % "{0:.3%}".format(accuracy))

    kf = KFold(n_splits=5)
    error = []
    for train, test in kf.split(dataSet[X]):
        # Filter training data
        train_predictors = (dataSet[X].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = dataSet[Y].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(dataSet[X].iloc[test, :], dataSet[Y].iloc[test]))

    print("K FoldCross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))


def plot_learning_curve(estimator, title, X, Y, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, Y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    #plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
    #plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc="best")
    return plt



def generateReport(model,dataSet,X,Y):
    test_size = 0.33
    seed = 7
    X_train, X_test, Y_train, Y_test = train_test_split(dataSet[X], dataSet[Y], test_size=test_size,random_state=seed)
    model.fit(X_train, Y_train)
    accuracy=model.score(X_test,Y_test)
    print("Cross-Validation Score : %s" % "{0:.3%}".format(accuracy))
    predicted = model.predict(X_test)
    report = classification_report(Y_test, predicted)
    return report

