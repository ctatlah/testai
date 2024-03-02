'''
Created on Mar 2, 2024

@author: ctatlah
'''

class MyTest(object):
    def __init__(self, var1):
        self.myvar = var1
        
    def __str__(self):
        return self.myvar
    

if __name__ == '__main__':
    myc = MyTest("auto test")
    print(myc)
