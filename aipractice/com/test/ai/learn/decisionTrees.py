'''
Created on Mar 26, 2024

@author: ctatlah
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import com.test.ai.utils.dataUtils as dataUtil
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from mpmath import limit

#
# setup
#

#
# work
#

print ('Here we go, working with decision trees')

# Data
#
data = dataUtil.loadDataForDecisionTrees('tree_test_data.csv')

# do one-hot encoding
cat_variables = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
data = pd.get_dummies(data = data,
                         prefix = cat_variables,
                         columns = cat_variables)

features = [x for x in data.columns if x not in 'HeartDisease'] # choose all columns expect for 1
xTrain, xValidation, yTrain, yValidation = train_test_split(data[features], data['HeartDisease'], train_size = 0.8, random_state = 55)

# working with random forest
#
print('Random forest example')
minSamplesSplitList = [2, 10, 30, 50, 100, 200, 300, 700]
maxDepthList = [2, 4, 8, 16, 32, 64, None]
nEstimatorsList = [10, 50, 100, 500]

accuracyListTrain = []
accuracyListValidation = []

for minSamplesSplit in minSamplesSplitList:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(min_samples_split = minSamplesSplit,
                                   random_state = 55).fit(xTrain,yTrain) 
    predictionsTrain = model.predict(xTrain) ## The predicted values for the train dataset
    predictionsValidation= model.predict(xValidation) ## The predicted values for the test dataset
    accuracyTrain = accuracy_score(predictionsTrain,yTrain)
    accuracyValidation = accuracy_score(predictionsValidation,yValidation)
    accuracyListTrain.append(accuracyTrain)
    accuracyListValidation.append(accuracyValidation)

plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(minSamplesSplitList )),labels=minSamplesSplitList) 
plt.plot(accuracyListTrain)
plt.plot(accuracyListValidation)
plt.legend(['Train','Validation'])
plt.show()


# working with XGBoost
#
print('XGBoost examples')
print('example1:')
n = int(len(xTrain)*0.8) # Let's use 80% to train and 20% to eval
xTrainFit, xTrainEval, yTrainFit, yTrainEval = xTrain[:n], xTrain[n:], yTrain[:n], yTrain[n:]

xgbModel = XGBClassifier(n_estimators=500, learning_rate=0.1, verbosity=1, random_state=55)
xgbModel.fit(xTrainFit,yTrainFit, eval_set = [(xTrainEval, yTrainEval)], early_stopping_rounds=10)

xgbModel.best_iteration
print(f'XGBoost model best iteration = {xgbModel.best_iteration}')
print(f'Metrics train:\tAccuracy score = {accuracy_score(xgbModel.predict(xTrain),yTrain):.4f}')
print(f'Metrics test:\tAccuracy score = {accuracy_score(xgbModel.predict(xValidation),yValidation):.4f}')

# generated data that represents mushrooms. Cap Color Brown | Stalk Shape Tapered | Solitary | Edible
print('example2:')
xGenData = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0],
                     [1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0],
                     [1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
yGenData = np.array([1,1,0,0,1,0,0,1,1,0,
                     1,1,0,0,1,0,0,1,1,0,
                     1,1,0,0,1,0,0,1,1,0])

limit = int(len(xGenData)*0.8)
xGenTrain, yGenTrain, xGenTest, yGenTest = xGenData[:limit], yGenData[:limit], xGenData[limit:], yGenData[limit:]
limit = int(len(xGenTrain)*0.8)
xGenFit, yGenFit, xGenEval, yGenEval = xGenTrain[:limit], yGenTrain[:limit], xGenTrain[limit:], yGenTrain[limit:]

xgbGenModel = XGBClassifier(n_estimators=50, learning_rate=0.1, verbosity=1, random_state=55)
xgbGenModel.fit(xGenFit,yGenFit, eval_set = [(xGenEval, yGenEval)], early_stopping_rounds=10)
xgbGenModel.best_iteration
print(f'XGBoost model best iteration = {xgbGenModel.best_iteration}')
print(f'Metrics train:\tAccuracy score = {accuracy_score(xgbGenModel.predict(xGenData),yGenData):.4f}')
print(f'Metrics test:\tAccuracy score = {accuracy_score(xgbGenModel.predict(xGenTest),yGenTest):.4f}')

