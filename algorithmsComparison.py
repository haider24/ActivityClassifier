from sklearn.neural_network import MLPClassifier
import helper
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import matplotlib.pyplot as plt

if __name__ == "__main__":

    dataSet=helper.getData()

    Y = 'Activity'
    X = ['X', 'Y', 'Z']

    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('MLP', MLPClassifier()))
   # models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    algorithms = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        crossValidation = model_selection.cross_val_score(model, dataSet[X], dataSet[Y], cv=kfold, scoring=scoring)
        results.append(crossValidation)
        algorithms.append(name)
        meanAndStd = "%s: %f (%f)" % (name, crossValidation.mean(), crossValidation.std())
        print(meanAndStd)

    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(algorithms)
    plt.show()


