'''
Created on Mar 29, 2024

@author: ctatlah
'''
#
# imports
#
import time
import numpy as np
import tensorflow as tf
import com.test.ai.utils.dataUtils as dataUtil
import com.test.ai.utils.recommendUtils as recUtil
from tensorflow import keras #@UnresolvedImport

#
# setup
#
def mockNewUsersRatings(movieListDataFile, numMovies):
    newUserRatings = np.zeros(numMovies)          #  Initialize new users ratings

    # Check the file small_movie_list.csv for id of each movie in our dataset
    # For example, Toy Story 3 (2010) has ID 2700, so to rate it "5", you can set
    newUserRatings[2700] = 5 
    
    #Or suppose you did not enjoy Persuasion (2007), you can set
    newUserRatings[2609] = 2;
    
    # We have selected a few movies we liked / did not like and the ratings we gave are as follows:
    newUserRatings[929]  = 5   # Lord of the Rings: The Return of the King, The
    newUserRatings[246]  = 5   # Shrek (2001)
    newUserRatings[2716] = 3   # Inception
    newUserRatings[1150] = 5   # Incredibles, The (2004)
    newUserRatings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
    newUserRatings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
    newUserRatings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
    newUserRatings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
    newUserRatings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
    newUserRatings[2937] = 1   # Nothing to Declare (Rien à déclarer)
    newUserRatings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
    newUserRatedMovies = [i for i in range(len(newUserRatings)) if newUserRatings[i] > 0]
    
    print('\nNew user ratings:')
    for i in range(len(newUserRatings)):
        if newUserRatings[i] > 0 :
            print(f'Rated {newUserRatings[i]} for  {movieListDataFile.loc[i,"title"]}');
    
    print()
    return newUserRatings, newUserRatedMovies

#
# Work
#

print ('Here we go, lets recommend some stuff')

# data
#
X, W, b, numMovies, numFeatures, numUsers = dataUtil.loadDataForRecomendationsPreCalc()
Y, R = dataUtil.loadDataForRecomendationsRatings()
print('Data Summary:')
print('Y: ', Y.shape)
print('R: ', R.shape)
print('X: ', X.shape)
print('W: ', W.shape)
print('b: ', b.shape)
print('num features: ', numFeatures)
print('num movies: ', numMovies)
print('num users: ', numUsers)
print()

tsmean =  np.mean(Y[0, R[0, :].astype(bool)])
print(f'Average rating for movie 1 : {tsmean:0.3f} / 5')

startTime = time.time()
JSlow = recUtil.colabFilterCostFunc(X, W, b, Y, R, 1.5);
endTime = time.time()
print(f"Cost (with regularization): {JSlow}\n   runtime = {endTime - startTime} secs")

startTime = time.time()
JFast = recUtil.colabFilterCostFuncFast(X, W, b, Y, R, 1.5);
endTime = time.time()
print(f'Cost (with regularization) Fast Func: {JFast}\n   runtime = {endTime - startTime} secs')

movieList, movieListDataFile = dataUtil.loadDataForRecommendationsMovieList()
userRatings, userRated = mockNewUsersRatings(movieListDataFile, numMovies)

# learn movie recommendations for new user
#
Y, R = dataUtil.loadDataForRecomendationsRatings()
# Add new user ratings to Y 
Y = np.c_[userRatings, Y]
# Add new user indicator matrix to R
R = np.c_[(userRatings != 0).astype(int), R]
# Normalize the Dataset
Ynorm, Ymean = recUtil.normalizeRatings(Y, R)
numMovies, numUsers = Y.shape
numFeatures = 100
iterations = 200
lambda_ = 1

# Train model
#

# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((numUsers,  numFeatures), dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((numMovies, numFeatures), dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,         numUsers),    dtype=tf.float64),  name='b')

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

for iter in range(iterations):
    # Use TensorFlow’s GradientTape
    # to record the operations used to compute the cost 
    with tf.GradientTape() as tape:

        # Compute the cost (forward pass included in cost)
        costValue = recUtil.colabFilterCostFuncFast(X, W, b, Ynorm, R, lambda_)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads = tape.gradient(costValue, [X,W,b])

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, [X,W,b]))

    # Log periodically.
    if iter % 20 == 0:
        print(f'Training loss at iteration {iter}: {costValue:0.1f}')
        
# Make a prediction using trained weights and biases
#
prediction = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
predictionMean = prediction + Ymean # restore the mean
usersRatingPredictions = predictionMean[:,0]
sortedPredictions = tf.argsort(usersRatingPredictions, direction='DESCENDING')

# Output predictions performance
print('\nPredicted ratings:')
for i in range(17):
    j = sortedPredictions[i]
    if j not in userRated:
        print(f'Predicting rating {usersRatingPredictions[j]:0.2f} for movie {movieList[j]}')

print('\nOriginal vs Predicted ratings:')
for i in range(len(userRatings)):
    if userRatings[i] > 0:
        print(f'Original {userRatings[i]}, Predicted {usersRatingPredictions[i]:0.2f} for {movieList[i]}')

# Save to file
# filterRatings=(movieListDataFile['number of ratings'] > 20)
# movieListDataFile['pred'] = usersRatingPredictions
# movieListDataFile = movieListDataFile.reindex(columns=['pred', 'mean rating', 'number of ratings', 'title'])
# movieListDataFile.loc[sortedPredictions[:300]].loc[filterRatings].sort_values('mean rating', ascending=False)