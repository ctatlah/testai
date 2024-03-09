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
from sklearn.datasets import make_blobs 
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




'''
Multi class predictions
'''
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
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
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
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
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
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
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