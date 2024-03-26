'''
Created on Mar 3, 2024

@author: ctatlah

Wrapper for DataLoader to load test and training data for learning examples
'''

from com.test.ai.data.DataLoader import LoadData
from sklearn.model_selection import train_test_split

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
    
    x, y = loadData.readDataNpy('test_data_visual_predict_x.npy', 
                                'test_data_visual_predict_y.npy')
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
    return loadData.readDataNpy('test_data_visual_predict_x.npy', 
                                'test_data_visual_predict_y.npy')
    
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
    x, y = loadData.readDataFromCsv(filename)
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
