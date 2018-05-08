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
    allFiles = glob.glob(os.path.join(path, "*.txt"))
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
    score = []
    for train, test in kf.split(dataSet[X]):
        trainX = (dataSet[X].iloc[train, :])
        trainY = dataSet[Y].iloc[train]
        model.fit(trainX, trainY)
        score.append(model.score(dataSet[X].iloc[test, :], dataSet[Y].iloc[test]))

    print("K FoldCross-Validation Score : %s" % "{0:.3%}".format(np.mean(score)))


def plot_learning_curve(estimator, title, X, Y, cv=None,n_jobs=1, trainSizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    trainSizes, trainScores, testScores = learning_curve(estimator, X, Y, cv=cv, n_jobs=n_jobs, train_sizes=trainSizes)
    trainScoresMean = np.mean(trainScores, axis=1)
    testScoresMean = np.mean(testScores, axis=1)
    plt.grid()
    plt.plot(trainSizes, trainScoresMean, 'o-', color="r",label="Training score")
    plt.plot(trainSizes, testScoresMean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc="best")
    return plt



def generateReport(model,dataSet,X,Y):
    test_size = 0.33
    seed = 7
    XTrain, XTest, YTrain, YTest = train_test_split(dataSet[X], dataSet[Y], test_size=test_size,random_state=seed)
    model.fit(XTrain, YTrain)
    accuracy=model.score(XTest,YTest)
    print("Cross-Validation Score : %s" % "{0:.3%}".format(accuracy))
    predicted = model.predict(XTest)
    report = classification_report(YTest, predicted)
    return report

