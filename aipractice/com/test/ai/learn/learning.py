'''
Created on Mar 2, 2024

@author: ctatlah
'''
#
# imports
#
import numpy as np
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

# data
label = "test"
x = np.array([[200,20],
             [150,15],
             [275,17],
             [200,17]])
y = np.array([[1],
              [0],
              [0],
              [1]])
print (f"here we go with the *{label}* app")

# setup model
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tfk.Input(shape=(2,)), 
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
) 
model.summary()

# run model
model.compile(
    loss = tfk.losses.BinaryCrossentropy(),
    optimizer = tfk.optimizers.Adam(learning_rate=0.01), 
)

model.fit(
    x,y,            
    epochs=10,
)

# predict
X_test = np.array([
    [200,13.9],  # positive example
    [150,17]])   # negative example
predictions = model.predict(X_test) # raw prediction
print("predictions = \n", predictions)
yhat = (predictions >= 0.5).astype(int) # readable prediction category
print(f"decisions = \n{yhat}")