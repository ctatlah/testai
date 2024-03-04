'''
Created on Mar 3, 2024

@author: ctatlah

Wrapper for DataLoader to load test and training data for learning examples
'''

from com.test.ai.data.DataLoader import LoadData

def loadDataForNumberPerdiction():
    '''
    Loads visual number data (0 or 1) to test number predictions
    Args:
      none
    Return
      x (narray(n)) : visual data data (picture of 0 or 1)
      y (narray(n)) : actual value of that visual data (0 or 1)
    '''
    loadData = LoadData()
    return loadData.readDataNpy('test_data_visual_predict_x.npy', 'test_data_visual_predict_y.npy')