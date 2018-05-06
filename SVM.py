import helper
from sklearn.svm import SVC
from sklearn import svm
import matplotlib.pyplot as plt


if __name__ == "__main__":

    dataSet=helper.getData()
    X = ['X', 'Y', 'Z']
    Y = 'Activity'

    C = 1.0  # SVM regularization parameter
    #model = svm.SVC(kernel='linear', C=C)
    model=lin_svc = svm.LinearSVC(C=C)
    #model= svm.SVC(kernel='rbf', gamma=0.7, C=C)
    #model=poly_svc = svm.SVC(kernel='poly', degree=3, C=C)
    helper.testModel(model, dataSet, X, Y)

    report=helper.generateReport(model,dataSet,X,Y)
    print(report)
    title = "Learning Curves SVM"
    helper.plot_learning_curve(model, title, dataSet[X], dataSet[Y], cv=10, n_jobs=4)
    plt.show()
