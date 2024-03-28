'''
Created on Mar 27, 2024

@author: ctatlah
'''
#
# imports
#
import numpy as np
import matplotlib.pyplot as plt
import com.test.ai.utils.dataUtils as dataUtil
import com.test.ai.utils.aiUtils as aiUtil
import com.test.ai.utils.visualUtils as visUtil

#
# work
#

print ('Here we go, going to try to detect anomolies')

# data
#
xTrain, xCV, yCV = dataUtil.loadDataForAnomolyDetection(
    'anomoly_test_x_data1.npy', 
    'anomoly_test_x_val1.npy', 
    'anomoly_test_y_val1.npy')

print('xtrain data:')
print(xTrain)

# visualize data
plt.scatter(xTrain[:, 0], xTrain[:, 1], marker='x', c='b') 
plt.title("The first dataset")
plt.ylabel('Throughput (mb/s)')
plt.xlabel('Latency (ms)')
plt.axis([0, 30, 0, 30])
plt.show()

# calculate mean of every feature and variance
mu, var = aiUtil.estimate_gaussian(xTrain)              
print(f'Mean of each feature: {mu}')
print(f'Variance of each feature: {var}')

# getting the probability of each row in xTrain (density estimation)
# that will be compared to epsilon threshold
p = aiUtil.multivariate_gaussian(xTrain, mu, var)
print('p:')
print(p)
visUtil.visualize_anomoly_fit(xTrain, mu, var)

pCV = aiUtil.multivariate_gaussian(xCV, mu, var)
print('pCV:')
print(pCV)
epsilon, F1 = aiUtil.select_anomoly_threshold(yCV, pCV)

print(f'Best epsilon found using cross-validation: {epsilon}')
print(f'Best F1 on Cross Validation Set: {F1}')

# find the anomolies
outliers = p < epsilon
print('Visualize fit with anomolies circled')
visUtil.visualize_anomoly_fit(xTrain, mu, var)
plt.plot(xTrain[outliers, 0],  # Draw a red circle around those outliers
         xTrain[outliers, 1], 
         'ro', 
         markersize= 10,
         markerfacecolor='none', 
         markeredgewidth=2)
plt.show()



# do it with high dimentional dataset
#

print('Now lets do it on a high dimentional dataset')

# data
#
xTrainHigh, xCVHigh, yCVHigh = dataUtil.loadDataForAnomolyDetection(
    'anomoly_test_x_data2.npy', 
    'anomoly_test_x_val2.npy', 
    'anomoly_test_y_val2.npy')

# visualize data
plt.scatter(xTrainHigh[:, 0], xTrainHigh[:, 1], marker='x', c='b') 
plt.title('Multi dimentional dataset')
plt.ylabel('Throughput (mb/s)')
plt.xlabel('Latency (ms)')
plt.axis([0, 30, 0, 30])
plt.show()

# evaluate
muHigh, varHigh = aiUtil.estimate_gaussian(xTrainHigh)
pHigh = aiUtil.multivariate_gaussian(xTrainHigh, muHigh, varHigh)
pCVHigh = aiUtil.multivariate_gaussian(xCVHigh, muHigh, varHigh)
epsilonHigh, F1High = aiUtil.select_anomoly_threshold(yCVHigh, pCVHigh)
print(f'Mean of each feature: {muHigh}')
print(f'Variance of each feature: {varHigh}')
print(f'Best epsilon found using cross-validation: {epsilonHigh}')
print(f'Best F1 on Cross Validation Set:  {F1High}')
print(f'# Anomalies found: {sum(pHigh < epsilonHigh)}')
outliersHigh = pHigh < epsilonHigh
print(f'These are the anomolies:\n{outliersHigh}')
