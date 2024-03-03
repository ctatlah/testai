'''
Created on Mar 2, 2024

@author: ctatlah
'''

#import com.test.ai.my.MyTest
#from com.test.ai.my.MyTest import MyTest
#from com.test.ai.my.MyTest import MyTest as mt

from com.test.ai.data.DataLoader import LoadData


#myTest = com.test.ai.my.MyTest.MyTest("hello world1")
#print(myTest)
#myTest = MyTest("hello world2")
#print(myTest)
#myTest = MyTest("hello world3")
#print(myTest)
#myTest = mt("hello world4")
#print(myTest)


loadData = LoadData()
data = loadData.read('test_data.txt')
print(f'printing data...\n{data}')

xdata, ydata = loadData.readTrainingData('test_data_ai.txt')
print(f'printing xdata...\n{xdata} \nprinting ydata...\n{ydata}')

