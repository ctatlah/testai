'''
Created on Mar 2, 2024

@author: ctatlah
'''
#
# imports
#
import com.test.ai.utils.visualUtils as visUtil
import com.test.ai.utils.dataUtils as dataUtil
import tensorflow as tf
import tensorflow.keras as tfk #@UnresolvedImport
from tensorflow.keras.models import Sequential #@UnresolvedImport
from tensorflow.keras.layers import Dense #@UnresolvedImport

import logging

#
# setup
#
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

#
# work
#

print ('Here we go, going to try to read numbers')

# data
#
x, y = dataUtil.loadDataForNumberPerdiction()


# setup model
#   Layer1 : 25 neurons
#   Layer2 : 15 neurons
#   Layer3 : 1 neuron (output layer)
#
print('Setting up model...')
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
print(f'Predicting a zero: {prediction}' )

prediction = model.predict(x[500].reshape(1,400))  # a one
print(f'Predicting a one:  {prediction}')
prediction = 1 if prediction >= 0.5 else 0
print(f'Predicting a one: {prediction}' )


# visualize prediction
#
print ('Visualize prediction data...')
visUtil.visualizeNumberPrediction(x, y, model)
