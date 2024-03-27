'''
Created on Mar 3, 2024

@author: ctatlah
'''

import numpy as np
import matplotlib.pyplot as plt
import com.test.ai.utils.aiUtils as aiUtil

def visualizeNumberPrediction(x, y, model, predictLogic):
    '''
    Creates a plt to visualize number reading data
    Args:
      x (narray(n)) : data to plot
      y (narray(n)) : expected data
      model (Sequential) : tenserflow sequential neural network model
      predictLogic (func(d,m) : logic to use to make prediction
              d -- data at some random index
              m -- neural network model
    Output:
      Plot of visual number data
    Borrowed code from deeplearning.ai and modified slightly
    '''
    m, n = x.shape

    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]
    
    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and reshape the image
        x_random_reshaped = x[random_index].reshape((20,20)).T
        
        # Display the image
        ax.imshow(x_random_reshaped, cmap='gray')
        
        # Predict using the Neural Network
        prediction = predictLogic(x[random_index], model)
        
        # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{prediction}")
        ax.set_axis_off()
    fig.suptitle("Legend: Expected, My Prediction", fontsize=16)
    plt.show()

def visualize_anomoly_fit(X, mu, var):
    '''
    This visualization shows you the 
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    '''
    
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = aiUtil.multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), linewidths=1)
        
    # Set the title
    plt.title("The Gaussian contours of the distribution fit to the dataset")
    # Set the y-axis label
    plt.ylabel('Throughput (mb/s)')
    # Set the x-axis label
    plt.xlabel('Latency (ms)')