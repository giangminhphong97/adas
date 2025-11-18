import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

############ UTILITY FUNCTION #############
def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)

    # Retrieve the number of color channels of the image.
    # channel_count = img.shape[2] # Used only for ROI of original image

    # Create a match color with the same color channel counts.
    # match_mask_color = (255,) * channel_count # Used only for ROI of original image
    match_mask_color = (255,)

    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (img.shape[0], img.shape[1], 3),
        dtype=np.uint8,
    )
    # Make a copy of the original image.
    img = np.copy(img)

    # If there are no lines to draw, exit.
    if lines is None:
        return

    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    # Return the modified image.
    return img

############# IMAGE PROCESSING #############
def pipeline(image):
    height = image.shape[0]
    width = image.shape[1]

    # Cropping to a region of interest
    region_of_interest_vertices = [
        (0, 590),
        (1052 / 2, 590 / 2),
        (1052, 590),
    ]

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny Edge Detection here
    cannyed_image = cv2.Canny(gray_image, 100, 200)

    cropped_image = region_of_interest(cannyed_image,np.array([region_of_interest_vertices], np.int32), )

    lines = cv2.HoughLinesP(cropped_image,
                            rho=6, 
                            theta=np.pi / 60,
                            threshold=160,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=25)

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope < 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    min_y = int(image.shape[0] * (3 / 5)) # Just below the horizon
    max_y = image.shape[0]

    poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    # line_image = draw_lines(image, lines)
    line_image = draw_lines(
        image,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
        thickness=5,
    )

    return line_image

# Reading in an image
image = mpimg.imread('path_to_image.jpg')
pipeline(image)

############# DISPLAY RESULTS #############
# Printing out some stats and plotting the image
print('This image is:', type(image), 'with dimensions:', image.shape)
# Create a 2x2 subplot grid and use the first three axes for our images.
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
# Flatten the axes array for easy indexing (axs[0], axs[1], axs[2], axs[3])
axs = axs.flatten()

# Original image
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('on')

# Cropped image
axs[1].imshow(pipeline(image))
axs[1].set_title('Line Decection with Hough Transform')
axs[1].axis('on')

plt.tight_layout()
plt.show()