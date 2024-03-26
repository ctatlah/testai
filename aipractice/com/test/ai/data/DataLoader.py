'''
Created on Mar 3, 2024

@author: ctatlah
'''

import os
import numpy as np
import pandas as pd
from pathlib import Path

class LoadData(object):
    '''
    Loads data from a file
    '''

    def __init__(self):
        self.resFolder = Path('/Users/ctatlah/git/testai/aipractice/com/test/ai/resources')
      
    def read(self, filename):
        '''
        Loads data from file
        Args:
          filename (string) : Name of file
        Returns:
          data (ndarray (n)) : n rows of data from file
        '''
        data = []
        fileToOpen = self.resFolder / filename
        print(f'Reading data from file "{filename}": .', end='')
        numLines = 0
        with open(fileToOpen, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue # ignore empty line
                data.append(
                    list(np.fromstring(line, dtype=int, sep=","))
                )
                print('.',end='')
                numLines += 1
        print(f' Done. There were {numLines} lines read!')
        
        return data
    
    def readData(self, xfilename, yfilename):
        '''
        Loads data from file
        Args:
          xfilename (string) : filename for x data
          yfilename (string) : filename for y data
        Returns:
          xdata (ndarray (n)) : n rows of x data from file
          ydata (ndarray (n)) : m rows of y data from file
        '''
        datax = self.read(xfilename)
        datay = self.read(yfilename)
        return datax, datay
    
    def readDataNpy(self, xfilename, yfilename):
        '''
        Loads items from npy file
        Args:
          xfilename (string) : npy type filename for x data
          yfilename (string) : npy type filename for y data
        Returns:
          xdata (ndarray (n)) : n rows of x data from file
          ydata (ndarray (n)) : m rows of y data from file
        '''
        print(f'Loading x data from file "{xfilename}"...', end='')
        xfileToOpen = self.resFolder / xfilename
        X = np.load(xfileToOpen)
        print('Done!')
        print(f'Shape of x is : {str(X.shape)}')
        
        print(f'Loading y data from file "{yfilename}"...', end='')
        yfileToOpen = self.resFolder / yfilename
        y = np.load(yfileToOpen)
        print('Done!')
        print(f'Shape of y is : {str(y.shape)}')
        
        return X, y
    
    def readTrainingData(self, filename):
        '''
        Loads data from file for AI training
        Args:
          filename (string) : Name of file
        Returns:       
          xdata (ndarray (n)) : n rows of x data from file
          ydata (ndarray (m)) : m rows of y data from file
        '''
        datax = []
        datay = []
        fileToOpen = self.resFolder / filename
        print(f'Reading data from file "{filename}": .', end='')
        numXLines = 0
        numYLines = 0
        isXData = False
        with open(fileToOpen, 'r') as f:
            for line in f:
                line = line.strip()
                
                # checking if we are reading xdata or ydata then setting
                # appropriate flags
                if (isXData is not True) and (line == 'xdata'):
                    isXData = True
                    continue # this is xdata header skip reading header
                if (isXData is True) and (line == 'ydata'):
                    isXData = False
                    continue # this is ydata header skip reading header
                if line == '':
                    continue # ignore empty line
                
                # reading data and storing in correct place
                if isXData:
                    datax.append(
                        list(np.fromstring(line, dtype=int, sep=","))
                        )
                    numXLines += 1
                else:
                    datay.append(
                        list(np.fromstring(line, dtype=int, sep=","))
                        )
                    numYLines += 1
                print('.',end='')
        print(' Done!')
        print(f'There were {numXLines} lines of x data read.')
        print(f'There were {numYLines} lines of y data read.')
        
        return datax, datay
    
    def readDataFromCsv(self, filename):
        '''
        Loads items from csv file
        Args:
          filename (string) : file with csv data
        Returns:
          xdata (ndarray (n,1)) : n rows of x data from file
          ydata (ndarray (n,1)) : n rows of y data from file (last column from dataset)
        '''
        print(f'Reading data from file "{filename}"...', end='')
        fileToOpen = self.resFolder / filename
        data = np.loadtxt(fileToOpen, delimiter=',')
        print(' Done!')
        print(f'Shape of data is : {str(data.shape)}')
        
        # splitting x feature data and y expected results
        # expected should be the last column of the data set
        xdata = data[:,:-1]
        #xdata = np.expand_dims(xdata, axis=1) # 1-d array to 2-d array
        ydata = data[:,-1]
        #ydata = np.expand_dims(ydata, axis=1) # 1-d array to 2-d array
        
        return xdata, ydata
    
    def readCsv(self, filename):
        '''
        Uses pandas to loads data from csv file
        Args:
          filename (string) : file with csv data
        Returns:
          data : data from file
        '''
        print(f'Reading data from file "{filename}"...', end='')
        fileToOpen = self.resFolder / filename
        data = pd.read_csv(fileToOpen)
        print(' Done!')
        
        return data
    