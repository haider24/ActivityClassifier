from sklearn.naive_bayes import GaussianNB
import helper
import matplotlib.pyplot as plt


if __name__ == "__main__":

    dataSet=helper.getData()
    X = ['X', 'Y', 'Z']
    Y = 'Activity'

    model=GaussianNB()
    helper.testModel(model,dataSet,X,Y)
    report=helper.generateReport(model,dataSet,X,Y)
    print(report)
    title = "Learning Curves Naive Bayes"
    helper.plot_learning_curve(model, title, dataSet[X], dataSet[Y], cv=10, n_jobs=4)
    plt.show()
