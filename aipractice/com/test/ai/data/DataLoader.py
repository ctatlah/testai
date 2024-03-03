'''
Created on Mar 3, 2024

@author: ctatlah
'''

import numpy as np
import os
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
