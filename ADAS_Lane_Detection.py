import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from fct_ROI import region_of_interest

# Cropping to a region of interest
region_of_interest_vertices = [
    (0, 590),
    (1052 / 2, 590 / 2),
    (1052, 590),
]

# Reading in an image
image = mpimg.imread('path_to_image.jpg')

cropped_image = region_of_interest(image,
                                       np.array([region_of_interest_vertices], np.int32), )

# Printing out some stats and plotting the image
print('This image is:', type(image), 'with dimensions:', image.shape)
fig,axs=plt.subplots(1,2, figsize=(15,7))
axs[0].imshow(image) 
axs[0].set_title('Original Image')
axs[1].imshow(cropped_image)
axs[1].set_title('Cropped Image')
plt.show() 