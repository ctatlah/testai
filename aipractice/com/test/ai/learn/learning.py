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
import matplotlib.pyplot as plt
from com.test.ai.data.DataLoader import LoadData
import logging

#
# setup
#
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

#
# work
#

print ("here we go with the test app")

# data
#
loadData = LoadData()
x, y = loadData.readDataNpy('test_data_ai_x.npy', 'test_data_ai_y.npy')

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
