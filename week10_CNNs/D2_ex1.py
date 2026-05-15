import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve # SciPy lib to perform conv on image

# Load a sample grayscale image 
image = np.random.rand(10,10)

# Define convolutional kernels (filters)
# Edge detection kernel 
edge_kernel = np.array([
    [-1, -1, -1],
    [-1, -8, -1],
    [-1, -1, -1]
])

# Blur kernel 
blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])/9

# Apply convolution
edge_image = convolve(image, edge_kernel)
blur_image = convolve(image, blur_kernel)

# Visualize original and filterd images 
## Create subplots
fig, axes = plt.subplots(1, 3, figsize = (12,4))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Image")
axes[1].imshow(edge_image, cmap = "gray")
axes[1].set_title("Edged Detection Image")
axes[2].imshow(blur_image, cmap = "gray")
axes[2].set_title("Blurred Image")

plt.show()