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
import com.test.ai.utils.kMeansUtils as kMeansUtil

#
# Work
#

# data
#
image = dataUtil.loadImageData('bird_small.png')
print(f'shape of image:{image.shape}')
plt.imshow(image)
plt.show()

# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.
xImg = np.reshape(image, (image.shape[0] * image.shape[1], 3))

# K-means
#
K = 16
maxIters = 10

initialCentroids = kMeansUtil.kMeans_init_centroids(xImg, K)
centroids, idx = kMeansUtil.run_kMeans(xImg, initialCentroids, maxIters)
print(f'shape of idx:{idx.shape}')
print(f'closest centroid for the first five elements:{idx[:5]}')
kMeansUtil.plot_kMeans_RGB(xImg, centroids, idx, K)

# compress image
#
# Find the closest centroid of each pixel
idx = kMeansUtil.find_closest_centroids(xImg, centroids)

# Replace each pixel with the color of the closest centroid
xRecovered = centroids[idx,:] 

# Reshape image into proper dimensions
imageCompressed = np.reshape(xRecovered, image.shape)

# Display original image
fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')

ax[0].imshow(image)
ax[0].set_title('Original')
ax[0].set_axis_off()
plt.show()

# Display compressed image
ax[1].imshow(imageCompressed)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
plt.show()
 