import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

## Preprocessing
trainData = pd.read_csv("C:\\Users\\venka\\Desktop\\Kaggle\\Titanic\\train.csv")
testData = pd.read_csv("C:\\Users\\venka\\Desktop\\Kaggle\\Titanic\\test.csv")
testY = pd.read_csv("C:\\Users\\venka\\Desktop\\Kaggle\\Titanic\\submission.csv")

testX = testData.drop(['Name','Cabin','Ticket'],axis=1)
trainX = trainData.drop(['Name','Cabin','Survived','Ticket'],axis=1)
trainY = trainData['Survived']

trainX['Age'] = trainX['Age'].fillna(value=trainX['Age'].dropna().mean())
testX['Age'] = testX['Age'].fillna(value=testX['Age'].dropna().mean())
trainX['nSex'] = trainX.Sex.map({'male':1,'female':0})
testX['nSex'] = testX.Sex.map({'male':1,'female':0})
trainX['nEmbarked'] = trainX.Embarked.map({'S':2,'Q':1,'C':0})
testX['nEmbarked'] = testX.Embarked.map({'S':2,'Q':1,'C':0})
trainX = trainX.drop(['Sex','Embarked'],axis=1)
testX = testX.drop(['Sex','Embarked'],axis=1)

#print(trainX.describe())
#print(testX.describe())

## Logistic Regression
'''logisticRegression = sklearn.linear_model.LogisticRegression()
onehotEncoder = sklearn.preprocessing.OneHotEncoder()

logisticFit = logisticRegression.fit(trainX,trainY)
currentPreds = logisticRegression.predict(testX)'''


## Decision Trees
'''decisionTree = sklearn.tree.DecisionTreeClassifier()
dTreeFit = decisionTree.fit(trainX,trainY)
currentPreds = dTreeFit.predict(testX)'''


## Neural Nets - MLPClassifier
trainX = trainX[['nSex']]
testX = testX[['nSex']]
MLP = MLPClassifier(hidden_layer_sizes=1000,alpha=0.01)
MLPFit = MLP.fit(trainX,trainY)
trainPred = MLP.predict(trainX)
currentPreds = MLP.predict(testX)



## Prediction
conf = confusion_matrix(trainY,trainPred)
print(conf)
print("Accuracy: "+str((conf[0][0]+conf[1][1])/(conf[0][0]+conf[1][1]+conf[1][0]+conf[0][1])))
testY['Survived'] = currentPreds
testY.to_csv("C:\\Users\\venka\\Desktop\\Kaggle\\Titanic\\NN_submission_6.csv",index=False)

