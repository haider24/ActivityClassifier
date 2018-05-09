import pickle
import helper
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from prettytable import PrettyTable

def generateModels():
    dataSet=helper.getData()
    X = ['X', 'Y', 'Z']
    Y = 'Activity'
    knnClassifier=KNeighborsClassifier()
    knnClassifier.fit(dataSet[X],dataSet[Y])

    decisionTreeClassifier = DecisionTreeClassifier()
    decisionTreeClassifier.fit(dataSet[X], dataSet[Y])

    randomForestClassifier = RandomForestClassifier()
    randomForestClassifier.fit(dataSet[X], dataSet[Y])

    pickle.dump(knnClassifier,open('knnClassifier','wb'))
    pickle.dump(decisionTreeClassifier, open('decisionTreeClassifier', 'wb'))
    pickle.dump(randomForestClassifier, open('randomForestClassifier', 'wb'))

def loadModels():
    knnClassifier = pickle.load(open('knnClassifier', 'rb'))
    decisionTreeClassifier = pickle.load(open('decisionTreeClassifier', 'rb'))
    randomForestClassifier = pickle.load(open('randomForestClassifier', 'rb'))
    return knnClassifier,decisionTreeClassifier,randomForestClassifier

def loadTestData():
    testData=pd.read_csv('testData.txt', header=None, delimiter=r"\s+", names=["X", "Y", "Z","Activity"])
    return testData

if __name__ == "__main__":

    knnClassifier,decisionTreeClassifier,randomForestClassifier=loadModels()
    testData=loadTestData()
    testDataLabels=pd.DataFrame()
    testDataLabels['Activity']=testData['Activity']
    del testData['Activity']
    X = ['X', 'Y', 'Z']
    Y = 'Activity'
    table = PrettyTable(['X','Y','Z','KNN Predicted Label','Decision Tree Predicted Label','Random Forest Predicted Label','Actual Label'])

    for row in range(len(testData.index)):
        test=testData.iloc[row].values.reshape(1,-1)
        table.add_row([testData.loc[testData.index[row],'X'],testData.loc[testData.index[row],'Y'],testData.loc[testData.index[row],'Z'],knnClassifier.predict(test)[0], decisionTreeClassifier.predict(test)[0], randomForestClassifier.predict(test)[0],testDataLabels.iloc[row][0]])

    print(table)