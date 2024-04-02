'''
Created on Mar 3, 2024

@author: ctatlah

Wrapper for DataLoader to load test and training data for learning examples
'''
import csv
import pickle
import pandas as pd
from com.test.ai.data.DataLoader import LoadData
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from collections import defaultdict


loadData = LoadData()

def loadDataForNumberPerdiction():
    '''
    Loads visual number data (0 or 1) to test number predictions
    Args:
      none
    Return
      x (narray(n)) : visual data data (picture of 0 or 1)
      y (narray(n)) : actual value of that visual data (0 or 1)
    '''
    
    print('Loading x data:')
    x = loadData.readDataNpy('test_data_visual_predict_x.npy')
    
    print('Loading y data:') 
    y = loadData.readDataNpy('test_data_visual_predict_y.npy')
    
    x = x[0:1000]
    y = y[0:1000]
    
    return x, y

def loadDataForFullNumberPerdiction():
    '''
    Loads visual number data (0 through 9) to test number predictions
        Args:
          none
        Return
          x (narray(n)) : visual data data (picture of digits from 0 through 9)
          y (narray(n)) : actual value of that visual data (0 through 9)
    '''
    print('Loading x data:')
    x = loadData.readDataNpy('test_data_visual_predict_x.npy') 
    
    print('Loading y data:')
    y = loadData.readDataNpy('test_data_visual_predict_y.npy')
    
    return x, y

def loadDataForModelEvaluations(filename):
    '''
    Loads data from a csv and splits it into training, cross validation, and test data sets.
    60% training, 20% cross validation, 20% test
        Args:
          fileName : CSV file to load data from
        Return
          xTrain (narray(n)) : x training data
          yTrain (narray(n)) : y training data
          xCrossValidation (narray(n)) : x cv data
          yCrossValidation (narray(n)) : y cv data
          xTest (narray(n)) : x test data
          yTest (narray(n)) : y test data
    '''
    d = loadData.readDataFromCsv(filename)
    
    # splitting x feature data and y expected results
    # expected should be the last column of the data set
    x = d[:,:-1]
    y = d[:,-1]
    
    # 60% data for training 40% for other
    xTrain, x_, yTrain, y_ = train_test_split(x, y, test_size=0.40, random_state=1)
    # split other 50% cv and 50% test 
    xCrossValidataion, xTest, yCrossValidation, yTest = train_test_split(x_, y_, test_size=0.50, random_state=1)
    
    del x_, y_
    
    return xTrain, yTrain, xCrossValidataion, yCrossValidation, xTest, yTest

def loadDataForDecisionTrees(filename):
    '''
    Loads data from CSV file to work with decision trees
        Args:
          fileName : CSV file to load data from
        Return
          data : data to work with
    '''
    return loadData.readCsv(filename)

def loadImageData(filename):
    '''
    Load an image
    Args:
      filename (string) : image file
    Returns:
      image : image data
    '''
    return loadData.loadImage(filename)

def loadDataForAnomolyDetection(xFilename, xCVFilename, yCVFilename):
    '''
    Load data for anomoly detection example
    Args:
      xFilename (string) : file name for x data
      xCVFilename (string) : file name for x cross validation data
      yCVFilename (string) : file name for y cross validation data
    Returns:
      x (ndarray) : x data
      xCrossValidation (ndarray) : x cross validation data
      yCrossValidation (ndarray) : y cross validation data
    '''
    print('Loading x train data:')
    x = loadData.readDataNpy(xFilename)
    print('Loading x cv data:')
    xCrossValidation = loadData.readDataNpy(xCVFilename)
    print('Loading y cv data:')
    yCrossValidation = loadData.readDataNpy(yCVFilename)
    
    return x, xCrossValidation, yCrossValidation

def loadDataForRecomendationsPreCalc():
    x = loadData.readDataFromCsv('small_movies_X.csv')
    w = loadData.readDataFromCsv('small_movies_W.csv')
    b = loadData.readDataFromCsv('small_movies_b.csv')
    b = b.reshape(1,-1)
    
    numMovies, numFeatures = x.shape
    numUsers,_ = w.shape
    
    return(x, w, b, numMovies, numFeatures, numUsers)

def loadDataForRecomendationsRatings():
    y = loadData.readDataFromCsv('small_movies_Y.csv')
    r = loadData.readDataFromCsv('small_movies_R.csv')
    return(y, r)

def loadDataForRecommendationsMovieList():
    ''' 
    Returns df with and index of movies in the order they are in in the Y matrix 
    '''
    filename = loadData.resFolder / 'small_movie_list.csv'
    dataFile = pd.read_csv(filename, header=0, index_col=0,  delimiter=',', quotechar='"')
    movieList = dataFile['title'].to_list()
    return(movieList, dataFile)

def loadDataForMovieRecommendationSystem():
    ''' 
    Load data for recommendationSystem2.py
    '''
    print(f'Reading data for Recommendation System from files...', end='')
    fileToOpenItemTrain = loadData.resFolder / 'content_item_train.csv'
    fileToOpenUserTrain = loadData.resFolder / 'content_user_train.csv'
    fileToOpenYTrain = loadData.resFolder / 'content_y_train.csv'
    fileToOpenItemTrainHeader = loadData.resFolder / 'content_item_train_header.txt'
    fileToOpenUserTrainHeader= loadData.resFolder / 'content_user_train_header.txt'
    fileToOpenItemVecs = loadData.resFolder / 'content_item_vecs.csv'
    fileToOpenMovieList = loadData.resFolder / 'content_movie_list.csv'
    fileToOpenUserToGenre = loadData.resFolder / 'content_user_to_genre.pickle'
    
    itemTrain = genfromtxt(fileToOpenItemTrain, delimiter=',')
    userTrain = genfromtxt(fileToOpenUserTrain, delimiter=',')
    yTrain = genfromtxt(fileToOpenYTrain, delimiter=',')
    
    # csv reader handles quoted strings better
    with open(fileToOpenItemTrainHeader, newline='') as f:    
        itemFeatures = list(csv.reader(f))[0]
    with open(fileToOpenUserTrainHeader, newline='') as f:
        userFeatures = list(csv.reader(f))[0]
    itemVecs = genfromtxt(fileToOpenItemVecs, delimiter=',')

    # Create movie dictionary
    movieDict = defaultdict(dict)
    count = 0
    with open(fileToOpenMovieList, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in reader:
            if count == 0:
                count += 1  #skip header
            else:
                count += 1
                movieId = int(line[0])
                movieDict[movieId]['title'] = line[1]
                movieDict[movieId]['genres'] = line[2]

    with open(fileToOpenUserToGenre, 'rb') as f:
        userToGenre = pickle.load(f)
        
    print('Done!')

    return(itemTrain, 
           userTrain, 
           yTrain, 
           itemFeatures, 
           userFeatures, 
           itemVecs, 
           movieDict, 
           userToGenre)

def loadDataForMovieRecommendationSystemMovieRatingDataSet():
    top10DataFile = loadData.readCsv('content_top10_df.csv')
    byGenreDataFile = loadData.readCsv('content_bygenre_df.csv')
    return(top10DataFile, byGenreDataFile)
