from sklearn.ensemble import RandomForestClassifier
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
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('MLP', MLPClassifier()))
    models.append(('RFC', RandomForestClassifier()))
   # models.append(('SVM', SVC()))
    results = []
    algorithms = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        crossValidation = model_selection.cross_val_score(model, dataSet[X], dataSet[Y], cv=kfold, scoring=scoring)
        results.append(crossValidation)
        algorithms.append(name)
        meanAndStd = "%s: %s" % (name, "{0:.3%}".format(crossValidation.mean()))
        print(meanAndStd)



    fig = plt.figure()
    fig.suptitle('Learning Algorithms Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(algorithms)
    plt.show()


