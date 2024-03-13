'''
Created on Mar 13, 2024

@author: ctatlah
'''
#
# imports
#
import numpy as np
import com.test.ai.utils.visualUtils as visUtil
import com.test.ai.utils.dataUtils as dataUtil
import tensorflow as tf
import tensorflow.keras as tfk #@UnresolvedImport
from tensorflow.keras.models import Sequential #@UnresolvedImport
from tensorflow.keras.layers import Dense #@UnresolvedImport
from tensorflow.keras.activations import linear, relu, sigmoid #@UnresolvedImport

import logging

#
# setup
#
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

#
# work
#

print ('Here we go, going to try to read full set of numbers')

# data
#
x, y = dataUtil.loadDataForFullNumberPerdiction()

# setup model
#   Layer1 : 25 neurons
#   Layer2 : 15 neurons
#   Layer3 : 10 neuron (output layer)
#
print('Setting up model...')
model = Sequential(
    [ 
        tfk.Input(shape=(400,)), # data shape
        Dense(25, activation = 'relu', name='layer1'),
        Dense(15, activation = 'relu', name='layer2'),
        Dense(10, activation = 'linear', name='layer3')
    ], name = 'full_number_read_model'
)
model.summary()

# run model
#
print ('Compile and Fit model...')
model.compile(
    loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tfk.optimizers.Adam(0.001),
)

model.fit(
    x,y,            
    epochs=50,
)

# predict
#
print ('Make some predictions...')
xToTest = x[1015]
prediction = model.predict(xToTest.reshape(1,400))
print(f'Predicting a Two: \n{prediction}')
print(f'Largest Prediction index: {np.argmax(prediction)}')
prediction_p = tf.nn.softmax(prediction)
print(f'Predicting a Two. Probability vector: \n{prediction_p}')
print(f'Total of predictions: {np.sum(prediction_p):0.3f}')
prediction_y = np.argmax(prediction_p)
print(f'np.argmax(prediction_p): {prediction_y}')

def predictFullNumbers(d, m):
    p = m.predict(d.reshape(1,400))
    p_prob = tf.nn.softmax(p)
    return np.argmax(p_prob)

# visualize prediction
#
print ('Visualize prediction data...')
visUtil.visualizeNumberPrediction(x, y, model, predictFullNumbers)