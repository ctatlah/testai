'''
Created on Mar 2, 2024

@author: ctatlah
'''

#import com.test.ai.my.MyTest
from com.test.ai.my import MyTest
#from com.test.ai.my import MyTest as mt

#myTest = com.test.ai.my.MyTest("hello world")
myTest = MyTest("hello world")
#myTest = mt("hello world")

print(myTest)