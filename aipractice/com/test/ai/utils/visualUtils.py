'''
Created on Mar 3, 2024

@author: ctatlah
'''

import numpy as np
import matplotlib.pyplot as plt

def visualizeNumberPrediction(x, y, model):
    '''
    Creates a plt to visualize number reading data
    Args:
      x (narray(n)) : data to plot
      y (narray(n)) : expected data
      model (Sequential) : tenserflow sequential neural network model
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
        prediction = model.predict(x[random_index].reshape(1,400))
        prediction = 1 if prediction >= 0.5 else 0
        
        # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{prediction}")
        ax.set_axis_off()
    fig.suptitle("Legend: Expected, My Prediction", fontsize=16)
    plt.show()

