'''
Created on Mar 21, 2024

@author: ctatlah
'''

#
# imports
#
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as tfk #@UnresolvedImport
from tensorflow.keras.models import Sequential #@UnresolvedImport
from tensorflow.keras.layers import Dense #@UnresolvedImport
import com.test.ai.utils.dataUtils as dataUtil
# for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
#from pip._vendor.rich.live import examples

#
# setup
#
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

#
# work
#

'''
Predictions with sklearn and evaluating model
'''
print ('here we go with linear regression with sklearn and evaluating model')

# data
# will have: 60% data for training
#            20% data for cross validation
#            20% data for test
#
xTrain, yTrain, xCV, yCV, xTest, yTest = dataUtil.loadDataForModelEvaluations('test_data_csv_train.csv')

# perform feature scaling to help model converge faster.
scalerLinear = StandardScaler()
xTrainScaled = scalerLinear.fit_transform(xTrain)

# create model
linearModel = LinearRegression()

# train model
linearModel.fit(xTrainScaled, yTrain)

print('\nevaluating linear regression trained model')
# predict
yhat = linearModel.predict(xTrain)
print(f'training mean squared error: {mean_squared_error(yTrain, yhat) / 2}')

# look at model with cross validation data
xCVScaled = scalerLinear.transform(xCV)
yhatCV = linearModel.predict(xCVScaled)
print(f'cross validation mean squared error: {mean_squared_error(yCV, yhatCV) / 2}')

# 
# now
# manipulate data adding more features (degree var)
# then create new model
#
print('\nnow manipulate data adding more features and re-evaluate model')
poly = PolynomialFeatures(degree=2, include_bias=False)
xTrainMapped = poly.fit_transform(xTrain)
scalerPoly = StandardScaler()
xTrainMappedScaled = scalerPoly.fit_transform(xTrainMapped)

# create model
model = LinearRegression()

# train model
model.fit(xTrainMappedScaled, yTrain )

# predict
yhatPoly = model.predict(xTrainMappedScaled)
print(f'polynomial training MSE: {mean_squared_error(yTrain, yhatPoly) / 2}')

# with cross validation data
xCVMapped = poly.transform(xCV)
xCVMappedScaled = scalerPoly.transform(xCVMapped)
yhatCVMapped = model.predict(xCVMappedScaled)
print(f'polynomial CV MSE: {mean_squared_error(yCV, yhatCVMapped) / 2}')
print('\n')

def helperTrainPolyLinearModels(model, xTrain, yTrain, xCV, yCV, maxDegree=10, baseline=None, regParams=None):
    '''
    helper function to help add additional features (polynomials--10) and train 10 models
    '''
    models = []
    trainMSES = []
    cvMSES = []
    scalers = []
    degrees = range(1,maxDegree+1)

    # Loop over 10 times. Each adding one more degree of polynomial higher than the last.
    for degree in degrees:

        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        xTrainMapped = poly.fit_transform(xTrain)

        # Scale the training set
        scalerPoly = StandardScaler()
        xTrainMappedScaled = scalerPoly.fit_transform(xTrainMapped)
        scalers.append(scalerPoly)

        # Create and train the model
        if regParams:
            model = Ridge(alpha=regParams)
        model.fit(xTrainMappedScaled, yTrain )
        models.append(model)

        # Compute the training MSE
        prediction = model.predict(xTrainMappedScaled)
        trainMSE = mean_squared_error(yTrain, prediction) / 2
        trainMSES.append(trainMSE)

        # Add polynomial features and scale the cross-validation set
        poly = PolynomialFeatures(degree, include_bias=False)
        xCVMapped = poly.fit_transform(xCV)
        xCVMappedScaled = scalerPoly.transform(xCVMapped)

        # Compute the cross-validation MSE
        predictionCV = model.predict(xCVMappedScaled)
        cvMSE = mean_squared_error(yCV, predictionCV) / 2
        cvMSES.append(cvMSE)
        
    # Plot the results
    xValues = regParams if regParams != None else degrees
    xLabel = 'lambda' if regParams != None else 'degree'
    titleLabel = 'lambda vs. train and CV MSEs' if regParams != None else 'degree of polynomial vs train and CV MSEs'
    plt.plot(xValues, trainMSES, marker='o', c='r', label='training MSEs'); 
    plt.plot(xValues, cvMSES, marker='o', c='b', label='CV MSEs'); 
    plt.plot(xValues, np.repeat(baseline, len(xValues)), linestyle='--', label='baseline')
    plt.title(titleLabel)
    plt.xticks(xValues)
    plt.xlabel(xLabel); 
    plt.ylabel('MSE'); 
    plt.legend()
    plt.show()
    
# more examples
#
print('here we go with some more evaluations')
xTrain, yTrain, xCV, yCV, xTest, yTest = dataUtil.loadDataForModelEvaluations('test_data_csv_train2.csv')
model = LinearRegression()
helperTrainPolyLinearModels(model, xTrain, yTrain, xCV, yCV, 15, baseline=400)

print('lets collect some more data and see how that works')
xTrain, yTrain, xCV, yCV, xTest, yTest = dataUtil.loadDataForModelEvaluations('test_data_csv_train3.csv')
modelForMoreData = LinearRegression()
helperTrainPolyLinearModels(modelForMoreData, xTrain, yTrain, xCV, yCV, 15, baseline=400)

print('try different lambdas')
regularParams = [10, 5, 2, 1, 0.5, 0.2, 0.1]
#helperTrainPolyLinearModels(modelForMoreData, xTrain, yTrain, xCV, yCV, 15, baseline=100, regParams=regularParams)



'''
Neural network finding the best number of nodes
'''
print ('here we go with finding best number of nodes in a neural network')

#data
xTrain, yTrain, xCV, yCV, xTest, yTest = dataUtil.loadDataForModelEvaluations('test_data_csv_train.csv')

# prep data
degree = 1
poly = PolynomialFeatures(degree, include_bias=False)
xTrainMapped = poly.fit_transform(xTrain)
xCVMapped = poly.transform(xCV)
xTestMapped = poly.transform(xTest)

scaler = StandardScaler()
xTrainMappedScaled = scaler.fit_transform(xTrainMapped)
xCVMappedScaled = scaler.transform(xCVMapped)
xTestMappedScaled = scaler.transform(xTestMapped)

# build test models
print('build 3 different neural nets to evaluate')
nnTrainMSES = [] # mean squared error train data
nnCVMSES = []    # mean squared error cv    data

tf.random.set_seed(20)

model1 = Sequential(
    [
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(1, activation = 'linear')
    ],
    name='model1'
)
model2 = Sequential(
    [
        Dense(20, activation = 'relu'),
        Dense(12, activation = 'relu'),
        Dense(12, activation = 'relu'),
        Dense(20, activation = 'relu'),
        Dense(1, activation = 'linear')
    ],
    name='model2'
)
model3 = Sequential(
    [
        Dense(32, activation = 'relu'),
        Dense(16, activation = 'relu'),
        Dense(8, activation = 'relu'),
        Dense(4, activation = 'relu'),
        Dense(12, activation = 'relu'),
        Dense(1, activation = 'linear')
    ],
    name='model3'
)
nnModels = [model1, model2, model3]

# loop over the the models
for model in nnModels:   
    # setup the loss and optimizer
    model.compile(
    loss='mse', # if binary yes/no problem --> loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer=tfk.optimizers.Adam(learning_rate=0.1),
    )
    print(f'training {model.name}...',end='')
    # train model
    model.fit(
        xTrainMappedScaled, yTrain,
        epochs=300,
        verbose=0
    )
    print('Done!')
    
    # record the training MSEs
    yhat = model.predict(xTrainMappedScaled)
    trainMSE = mean_squared_error(yTrain, yhat) / 2
    nnTrainMSES.append(trainMSE)
    
    # record the cross validation MSEs 
    yhat = model.predict(xCVMappedScaled)
    cvMSE = mean_squared_error(yCV, yhat) / 2
    nnCVMSES.append(cvMSE)
    
# print results
print('RESULTS:')
idealModelNum = 0
lowestCVMSES = 9999.9
for modelNum in range(len(nnTrainMSES)):
    print(
        f'Model {modelNum+1}: Training MSE: {nnTrainMSES[modelNum]:.2f}, ' +
        f'CV MSE: {nnCVMSES[modelNum]:.2f}'
        )
    currCVMSES = nnCVMSES[modelNum]
    if currCVMSES < lowestCVMSES:
        lowestCVMSES = currCVMSES
        idealModelNum = modelNum + 1

# now look at results and choose model with the lowest CV MSE which means
# you are using the best model and run it against the test data
yhat = nnModels[modelNum-1].predict(xTestMappedScaled)
testMSE = mean_squared_error(yTest, yhat) / 2
print(f'selected model: {idealModelNum}')
print(f'training MSE: {nnTrainMSES[idealModelNum-1]:.2f}')
print(f'cross validation MSE: {nnCVMSES[idealModelNum-1]:.2f}')
print(f'test MSE: {testMSE:.2f}')
print('\n')


