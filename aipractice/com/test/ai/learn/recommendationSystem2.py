'''
Created on Mar 29, 2024

@author: ctatlah
'''
#
# imports
#
import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk #@UnresolvedImport
import com.test.ai.utils.dataUtils as dataUtil
import com.test.ai.utils.recommendUtils as recUtil
import com.test.ai.utils.visualUtils as visUtil
from tensorflow import keras #@UnresolvedImport
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

#
# setup
#
pd.set_option("display.precision", 1)

def getNewUserVector():
    new_user_id = 5000
    new_rating_ave = 0.0
    new_action = 0.0
    new_adventure = 5.0
    new_animation = 0.0
    new_childrens = 0.0
    new_comedy = 0.0
    new_crime = 0.0
    new_documentary = 0.0
    new_drama = 0.0
    new_fantasy = 5.0
    new_horror = 0.0
    new_mystery = 0.0
    new_romance = 0.0
    new_scifi = 0.0
    new_thriller = 0.0
    new_rating_count = 3
    
    user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                          new_action, new_adventure, new_animation, new_childrens,
                          new_comedy, new_crime, new_documentary,
                          new_drama, new_fantasy, new_horror, new_mystery,
                          new_romance, new_scifi, new_thriller]])
    
    return user_vec

def genUserVecs(user_vec, num_items):
    ''' 
    given a user vector return: user predict maxtrix to match the size of item_vecs
    '''
    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs

#
# Work
#

print ('Here we go, lets recommend some stuff 2')

# data
#
top10Df, bygenreDf = dataUtil.loadDataForMovieRecommendationSystemMovieRatingDataSet()
itemTrain, userTrain, yTrain, itemFeatures, userFeatures, itemVecs, movieDict, userToGenre = dataUtil.loadDataForMovieRecommendationSystem()

print(f'\nTop 10 file contents:\n{top10Df}')
print(f'\nGenre file contents:\n{bygenreDf}')
print(f'Shape of item train: {itemTrain.shape}')
print(f'Shape of user train: {userTrain.shape}')
print(f'Shape of y train: {yTrain.shape}')
print(f'Length of item features: {len(itemFeatures)}')
print(f'Length of user features: {len(userFeatures)}')
print(f'Shape of item vecs: {itemVecs.shape}')

#numUserFeatures = userTrain.shape[1] - 3  # remove userid, rating count and ave rating during training
#numItemFeatures = itemTrain.shape[1] - 1  # remove movie id at train time
numUserFeatures = userTrain.shape[1]
numItemFeatures = itemTrain.shape[1]
_uvs = 3  # user genre vector start
_ivs = 3  # item genre vector start
#_us = 3  # start of columns to use in training, user
#_is = 1  # start of columns to use in training, items
print(f'Number of training vectors: {len(itemTrain)}')
print(f'Number item features: {numItemFeatures}')
print(f'Number user features: {numUserFeatures}')
#visUtil.recommendation_system_pprint_train(userTrain, userFeatures, _uvs,  _us, maxcount=5)

# scaling data
#
itemTrainUnscaled = itemTrain
userTrainUnscaled = userTrain
yTrainUnscaled    = yTrain

scalerItem = StandardScaler()
scalerItem.fit(itemTrain)
itemTrain = scalerItem.transform(itemTrain)

scalerUser = StandardScaler()
scalerUser.fit(userTrain)
userTrain = scalerUser.transform(userTrain)

scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(yTrain.reshape(-1, 1))
yTrain = scalerTarget.transform(yTrain.reshape(-1, 1))

np.allclose(itemTrainUnscaled, scalerItem.inverse_transform(itemTrain))
np.allclose(userTrainUnscaled, scalerUser.inverse_transform(userTrain))

itemTrain, itemTest = train_test_split(itemTrain, train_size=0.80, shuffle=True, random_state=1)
userTrain, userTest = train_test_split(userTrain, train_size=0.80, shuffle=True, random_state=1)
yTrain, yTest       = train_test_split(yTrain,    train_size=0.80, shuffle=True, random_state=1)
print(f"movie/item training data shape: {itemTrain.shape}")
print(f"movie/item test data shape: {itemTest.shape}")
print(f"user training data shape: {userTrain.shape}")
print(f"user test data shape: {userTest.shape}")
print(f"y training data shape: {yTrain.shape}")
print(f"y test data shape: {yTest.shape}")

# nerual network creation and train
#
print('creating NN')
numOutputs = 32
tf.random.set_seed(1)
userNN = tfk.models.Sequential([
    tfk.layers.Dense(256, activation='relu'),
    tfk.layers.Dense(128, activation='relu'),
    tfk.layers.Dense(numOutputs)
])

itemNN = tfk.models.Sequential([   
    tfk.layers.Dense(256, activation='relu'),
    tfk.layers.Dense(128, activation='relu'),
    tfk.layers.Dense(numOutputs) 
])

# create the user input and point to the base network
#inputUser = tfk.layers.Input(shape=(numUserFeatures,))
#vu = userNN(inputUser)
#vu = tf.linalg.l2_normalize(vu, axis=1)
inputUser = tfk.layers.Input(shape=(numUserFeatures,))
vu = userNN(inputUser)
vu = tfk.layers.Dense(numUserFeatures)(inputUser)
vuConcrete = tfk.backend.get_value(vu)

# got ValueError: A KerasTensor cannot be used as input to a TensorFlow function.
# with original implementation. Needed to wrap normilze function with custom one
class L2Normalize(tfk.layers.Layer):
    def call(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=1)

vu = L2Normalize()(inputUser)

# create the item input and point to the base network
# inputItem = tfk.layers.Input(shape=(numItemFeatures,))
# vm = itemNN(inputItem)
# vm = tf.linalg.l2_normalize(vm, axis=1)
inputItem = tfk.layers.Input(shape=(numItemFeatures,))
vm = itemNN(inputItem)
vm = tfk.layers.Dense(numItemFeatures)(inputItem)
vmConcrete = tfk.backend.get_value(vm)
vm = L2Normalize()(inputItem)

# compute the dot product of the two vectors vu and vm
output = tfk.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tfk.Model([inputUser, inputItem], output)

model.summary()

# setup cost function and compile model
tf.random.set_seed(1)
costFunc = tfk.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,loss=costFunc)

# train model
tf.random.set_seed(1)
#model.fit([userTrain[:, _us:], itemTrain[:, _is:]], yTrain, epochs=30)
model.fit([userTrain, itemTrain], yTrain, epochs=30)

# evaluate model against test data
#model.evaluate([userTest[:, _us:], itemTest[:, _is:]], yTest)
model.evaluate([userTest, itemTest], yTest)

# Prediction for new user
#
newUserVec = getNewUserVector()
userVecs = genUserVecs(newUserVec,len(itemVecs)) # generate and replicate the user vector to match the number movies in the data set.

# scale our user and item vectors
userVecsScaled = scalerUser.transform(userVecs)
itemVecsScaled = scalerItem.transform(itemVecs)

# make a prediction
#yp = model.predict([userVecsScaled[:, _us:], itemVecsScaled[:, _is:]])
yp = model.predict([userVecsScaled, itemVecsScaled])
print(f'Predictions for new user:\n{yp}')

# unscale y prediction 
ypUnscaled = scalerTarget.inverse_transform(yp)
print(f'Predictions for new user (Unscaled):\n{ypUnscaled}')

# sort the results, highest prediction first
sortedIndex = np.argsort(-ypUnscaled,axis=0).reshape(-1).tolist()  #negate to get largest rating first
sortedYPUnscaled   = ypUnscaled[sortedIndex]
sortedItems = itemVecs[sortedIndex]  #using unscaled vectors for display
table = recUtil.print_pred_movies(sortedYPUnscaled, sortedItems, movieDict, maxcount = 10)
print(table)



# Prediction for exisiting user
#
uid = 2 
# form a set of user vectors. This is the same vector, transformed and repeated.
userVecs, yVecs = recUtil.get_user_vecs(uid, userTrainUnscaled, itemVecs, userToGenre)

# scale our user and item vectors
userVecsScaled = scalerUser.transform(userVecs)
itemVecsScaled = scalerItem.transform(itemVecs)

# make a prediction
#yP = model.predict([suser_vecs[:, _us:], sitem_vecs[:, _is:]])
yP = model.predict([userVecsScaled, itemVecsScaled])
print(f'Predictions for existing user:\n{yP}')

# unscale y prediction 
yPU = scalerTarget.inverse_transform(yP)
print(f'Predictions for existing user (Unscaled):\n{yPU}')

# sort the results, highest prediction first
sortedIndex = np.argsort(-yPU,axis=0).reshape(-1).tolist()  #negate to get largest rating first
sortedYPU   = yPU[sortedIndex]
sortedItems = itemVecs[sortedIndex]  #using unscaled vectors for display
sortedUser  = userVecs[sortedIndex]
sortedY     = yVecs[sortedIndex]
table = recUtil.print_existing_user(sortedYPU, sortedY.reshape(-1,1), sortedUser, sortedItems, _ivs, _uvs, movieDict, maxcount = 50)
print(table)