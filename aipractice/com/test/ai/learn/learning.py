'''
Created on Mar 2, 2024

@author: ctatlah
'''
#
# imports
#
import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk #@UnresolvedImport
from tensorflow.keras.models import Sequential #@UnresolvedImport
from tensorflow.keras.layers import Dense #@UnresolvedImport
import matplotlib.pyplot as plt
from com.test.ai.data.DataLoader import LoadData
#import com.test.ai.utils.dataUtils as dataUtil
# for building linear regression models and preparing data
from sklearn.datasets import make_blobs 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#
# setup
#
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
loadData = LoadData()

#
# work
#

print ("here we go with the test app")


# data
#
x, y = loadData.readDataNpy('test_data_visual_predict_x.npy', 'test_data_visual_predict_y.npy')

print ('The shape of x is: ' + str(x.shape))
print ('The shape of y is: ' + str(y.shape))

# visualize data
#
print ('Visualize data...')
m, n = x.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1)

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and reshape the image
    x_random_reshaped = x[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(x_random_reshaped, cmap='gray')
    
    # Display the label above the image
    ax.set_title(y[random_index,0])
    ax.set_axis_off()

# setup model
#
model = Sequential(
    [
        tfk.Input(shape=(400,)), 
        Dense(25, activation='sigmoid', name='layer1'),
        Dense(15, activation='sigmoid', name='layer2'),
        Dense(1,  activation='sigmoid', name='layer3')
     ], name = 'test_model'
) 
model.summary()

# run model
#
print ('Compile and Fit model...')
model.compile(
    loss = tfk.losses.BinaryCrossentropy(),
    optimizer = tfk.optimizers.Adam(learning_rate=0.01), 
)

model.fit(
    x,y,            
    epochs=20,
)

# predict
#
print ('Make some predictions...')
prediction = model.predict(x[0].reshape(1,400))  # a zero
print(f'Predicting a zero: {prediction}')
prediction = 1 if prediction >= 0.5 else 0
print(f'Prediction a zero readable: {prediction}' )

prediction = model.predict(x[500].reshape(1,400))  # a one
print(f'Predicting a one:  {prediction}')
prediction = 1 if prediction >= 0.5 else 0
print(f'Prediction a one readable: {prediction}' )

# visualize perdiction
#
print ('Visualize prediction data...')
m, n = x.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and reshape the image
    x_random_reshaped = x[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(x_random_reshaped, cmap='gray')
    
    # Predict using the Neural Network
    prediction = model.predict(x[random_index].reshape(1,400))
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]},{yhat}")
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=16)
plt.show()




#############
# Multi class predictions
#############
print ('here we go with the test app for multi class predicitons')

# make up data
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

# model with softmax activation output layer
model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)
model.compile(
    loss=tfk.losses.SparseCategoricalCrossentropy(),
    optimizer=tfk.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs=10
)

prediction = model.predict(X_train)
print(prediction [:2])
print('largest value', np.max(prediction), 'smallest value', np.min(prediction))

# lets do it again but with more accuracy 
betterMultiCatModel = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear')
    ]
)
betterMultiCatModel.compile(
    loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tfk.optimizers.Adam(0.001),
)

betterMultiCatModel.fit(
    X_train,y_train,
    epochs=10
)

predictionBetter = betterMultiCatModel.predict(X_train)
print(f'two example output vectors:\n {predictionBetter[:2]}')
print('largest value', np.max(predictionBetter), 'smallest value', np.min(predictionBetter))

predictionBetterTFSM = tf.nn.softmax(predictionBetter).numpy()
print(f'two example output vectors:\n {predictionBetterTFSM[:2]}')
print('largest value', np.max(predictionBetterTFSM), 'smallest value', np.min(predictionBetterTFSM))
for i in range(5):
    print( f'{predictionBetter[i]}, category: {np.argmax(predictionBetter[i])}')
    
    
print ('\nnow we are doing it another way')
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)

# show classes in data set
print(f'unique classes {np.unique(y_train)}')
# show how classes are represented
print(f'class representation {y_train[:10]}')
# show shapes of our dataset
print(f'shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}')

# model
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        Dense(2, activation = 'relu',   name = 'layer1'),
        Dense(4, activation = 'linear', name = 'layer2')
    ]
)

model.compile(
    loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tfk.optimizers.Adam(0.01),
)

model.fit(
    X_train,y_train,
    epochs=200
)

l1 = model.get_layer('layer1')
W1,b1 = l1.get_weights()
print(f'weights for layer1')
print(f'W1: {W1}\n b1: {b1}')

l2 = model.get_layer('layer2')
W2, b2 = l2.get_weights()
print(f'weights for layer2')
print(f'W2: {W1}\n b2: {b1}')



#############
# Predictions with sklearn and evaluating model
#############
print ('here we go with linear regression with sklearn and evaluating model')

# data
# will have: 60% data for training
#            20% data for cross validation
#            20% data for test
#
x, y = loadData.readDataFromCsv('test_data_csv_train.csv')
# 60% data for training 40% for other
xTrain, x_, yTrain, y_ = train_test_split(x, y, test_size=0.40, random_state=1)
# split other 50% cv and 50% test 
xCV, xTest, yCV, yTest = train_test_split(x_, y_, test_size=0.50, random_state=1)
del x_, y_

# perform feature scaling to help model converge faster.
scalerLinear = StandardScaler()
xTrainScaled = scalerLinear.fit_transform(xTrain)


# create model
linearModel = LinearRegression()

# train model
linearModel.fit(xTrainScaled, yTrain)

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

#
# now you add additional polys and find the lowest MSE
# then add that poly to test data (degree=n 4,5,7,etc...) and predict + find mse
# **not doing this ran out of time today**
#



#############
# Neural network finding the best number of nodes
#############
print ('here we go with finding best number of nodes in a neural network')

#data
x, y = loadData.readDataFromCsv('test_data_csv_train.csv')
xTrain, x_, yTrain, y_ = train_test_split(x, y, test_size=0.40, random_state=1)
xCV, xTest, yCV, yTest = train_test_split(x_, y_, test_size=0.50, random_state=1)
del x_, y_

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
for modelNum in range(len(nnTrainMSES)):
    print(
        f'Model {modelNum+1}: Training MSE: {nnTrainMSES[modelNum]:.2f}, ' +
        f'CV MSE: {nnCVMSES[modelNum]:.2f}'
        )

# now look at results and choose model with the lowest CV MSE
# use the best model (for example model#3 will be different) against the test data
modelNum = 3
yhat = nnModels[modelNum-1].predict(xTestMappedScaled)
testMSE = mean_squared_error(yTest, yhat) / 2
print(f'selected model: {modelNum}')
print(f'training MSE: {nnTrainMSES[modelNum-1]:.2f}')
print(f'cross validation MSE: {nnCVMSES[modelNum-1]:.2f}')
print(f'test MSE: {testMSE:.2f}')



#############
# regularization on neural networks
#############
x, y = loadData.readDataFromCsv('test_data_csv_train2.csv')
xTrain, x_, yTrain, y_ = train_test_split(x, y, test_size=0.40, random_state=1)
xCV, xTest, yCV, yTest = train_test_split(x_, y_, test_size=0.50, random_state=1)
del x_, y_


nnLambda = 0.1
tf.random.set_seed(1234)
modelReg = Sequential(
    [
        Dense(120, activation = 'relu', kernel_regularizer=tfk.regularizers.l2(nnLambda)),
        Dense(40, activation = 'relu', kernel_regularizer=tfk.regularizers.l2(nnLambda)),
        Dense(1, activation = 'linear'),
    ], name='nnReg'
)
modelReg.compile(
    loss='mse',#tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tfk.optimizers.Adam(0.01),
)

modelReg.fit(
        xTrain,yTrain,
        epochs=1000
    )

# to loop through and find best lambda
lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
models=[None] * len(lambdas)
modelsMSEsForTrainingData = np.zeros(len(lambdas))
modelsMSEsForCVData = np.zeros(len(lambdas))
modelsMSEsForTestData = np.zeros(len(lambdas))

for i in range(len(lambdas)):
    lambda_ = lambdas[i]
    # create model
    models[i] =  Sequential(
        [
            Dense(120, activation = 'relu', kernel_regularizer=tfk.regularizers.l2(lambda_)),
            Dense(40, activation = 'relu', kernel_regularizer=tfk.regularizers.l2(lambda_)),
            Dense(1, activation = 'linear')
        ]
    )
    models[i].compile(
        loss='mse',#tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tfk.optimizers.Adam(0.01),
    )

    #train model
    models[i].fit(
        xTrain,yTrain,
        epochs=1000
    )
    
    #evaluate model errors
    predictionsOnTrainingData = models[i].predict(xTrain)
    trainMSE = mean_squared_error(yTrain, predictionsOnTrainingData) / 2
    modelsMSEsForTrainingData[i] = trainMSE
    print(f'model{i} @lambda_{lambda_} training error: {trainMSE}')
    
    predictionsOnCVData = models[i].predict(xCV)
    cvMSE = mean_squared_error(yCV, predictionsOnCVData) / 2
    modelsMSEsForCVData[i] = cvMSE
    print(f'model{i} @lambda_{lambda_} cv error: {cvMSE}')
    
    predictionsOnTestData = models[i].predict(xTest)
    testMSE = mean_squared_error(yTest, predictionsOnTestData) / 2
    modelsMSEsForTestData[i] = testMSE
    print(f'model{i} @lambda_{lambda_} cv error: {testMSE}')

print('REPORT:')
for i in range(len(lambdas)):
    print(f'model[{i}] @lambda_{lambdas[i]} training error: {modelsMSEsForTrainingData[i]:.2f}')
    print(f'model[{i}] @lambda_{lambdas[i]} cv error: {modelsMSEsForCVData[i]:.2f}')
    
#############
# 
#############