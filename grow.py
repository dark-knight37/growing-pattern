import cv2
from pdf2image import convert_from_path
import numpy as np

def add_region(new_region, search_region, x, y, e):    
    if new_region.count([x, y - e]) == 0:
        search_region.append([x, y - e])   
    if new_region.count([x, y + e]) == 0:
        search_region.append([x, y + e])   
    if new_region.count([x - e, y]) == 0:
        search_region.append([x - e, y])   
    if new_region.count([x + e, y]) == 0:
        search_region.append([x + e, y])
    return search_region

# Convert PDF to Image
pdf_path = "input.pdf"
images = convert_from_path(pdf_path, poppler_path='E:\Dev Tools\poppler\poppler-23.08.0\Library\exe')
image = images[0]
image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

new_region = []
search_region = []

# Define the center point and dimensions of the region and threshold
# center_x, center_y = 3000, 1800 #0.8
# center_x, center_y = 2100, 1800 #0.7
# center_x, center_y = 4400, 1200 #0.8
# center_x, center_y = 2800, 2700 #0.6
center_x, center_y = 3545, 3450 #0.165
# center_x, center_y = 5800, 3200 #0.6

tile_width = 100
e = 25
threshold = 0.8

# Calculate the top-left corner coordinates of the region
x = int(center_x - tile_width/2)
y = int(center_y - tile_width/2)

# Get subimage
subimg = image[center_y-int(tile_width/4):center_y+int(tile_width/4), center_x-int(tile_width/4):center_x+int(tile_width/4)]

# Get pattern
pattern = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Region growing
new_region.append([x, y])
search_region.append([x, y - e])
search_region.append([x, y + e])
search_region.append([x - e, y])
search_region.append([x + e, y])

while len(search_region):
    [cropped_x, cropped_y] = search_region.pop()
    cropped_region = image_gray[cropped_y:cropped_y+tile_width, cropped_x:cropped_x+tile_width]

    result = cv2.matchTemplate(cropped_region, pattern, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val > threshold:
        new_region.append([cropped_x, cropped_y])
        search_region = add_region(new_region, search_region, cropped_x, cropped_y, e)

print(len(new_region))
new_region = [list(x) for x in set(tuple(x) for x in new_region)]
print(len(new_region))

for [x, y] in new_region:
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x+tile_width, y+tile_width), (255,0,0), cv2.FILLED)

    # Transparency value
    alpha = 0.25

    # Perform weighted addition of the input image and the overlay
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# cv2.imshow('Detected Region', image)
cv2.imwrite('output.png', image)
cv2.imwrite('pattern.png', pattern)