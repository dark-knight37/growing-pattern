import cv2
from pdf2image import convert_from_path
import numpy as np

# Convert PDF to Image
pdf_path = "input.pdf"
images = convert_from_path(pdf_path, poppler_path='E:\Dev Tools\poppler\poppler-23.08.0\Library\exe')
image = images[0]
image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Define the center point and dimensions of the region
center_x = 2234
center_y = 1700
width = 100
height = 100
expansion_step = 50

# Calculate the top-left corner coordinates of the region
x = int(center_x - width/2)
y = int(center_y - height/2)

# Get subimage
subimg = image[y:y+height, x:x+width]

# Display the image
# cv2.imshow('Image', subimg)

pattern = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Region growing
top_left_y = center_y
bottom_right_y = center_y
top_left_x = center_x
bottom_right_x = center_x

top_left_y -= expansion_step
bottom_right_y += expansion_step
top_left_x -= expansion_step
bottom_right_x += expansion_step

cropped_region = image_gray[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

result = cv2.matchTemplate(cropped_region, pattern, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(max_val)

overlay = image.copy()
cv2.rectangle(overlay, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255,0,0), cv2.FILLED)

# Transparency value
alpha = 0.25

# Perform weighted addition of the input image and the overlay
result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# cv2.imshow('Detected Region', image)
cv2.imwrite('output.png', overlay)
cv2.imwrite('pattern.png', pattern)

cv2.waitKey(0)
cv2.destroyAllWindows()