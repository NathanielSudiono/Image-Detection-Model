import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image
image = cv2.imread('isePhoto7.jpg')
image = cv2.resize(image,(800,1050))

# Convert to grayscale
blur = cv2.GaussianBlur(image,(7,7), 1)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image
_, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

# erosion and dilation
kernel = np.ones((5,5), np.uint8)
#erosion = cv2.erode(thresh, kernel, iterations = 1)

edges = cv2.Canny(thresh, 15, 200)

dilation = cv2.dilate(edges, kernel, iterations = 2)



# Find contours
contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

triangle = 0
rectangular = 0
pentagon = 0
hexagon = 0
round = 0
total_object = 0

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 10000:
        total_object =total_object + 1
        epsilon = 0.04*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        cv2.drawContours(image, contour, -1, (255, 0, 255), 2)



        if len(approx) == 3:
            triangle = triangle + 1
            shape = "triangle"
        elif len(approx) == 4:
            rectangular = rectangular + 1
            shape = "rectangle"

        elif len(approx) == 5:
            rectangular = pentagon + 1
            shape = "Pentagon"

        elif len(approx) == 6:
            rectangular = hexagon + 1
            shape = "hexagon"

        else :
            round = round + 1
            shape = "round"

        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.putText(image, f"{shape} - Area: {area}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)




# Show the image
cv2.imshow('Image', image)
cv2.imshow('gray', gray)
cv2.imshow('threshold', thresh)
#cv2.imshow('erosion', erosion)
#cv2.imshow('dilation', dilation)

cv2.imshow('edges', edges)



print("triangular : ", triangle)
print("rectangular : ", rectangular)
print("pentagon : ", pentagon)
print("hexagon : ", hexagon)
print("round : ", round)
print("total objects :", total_object)

cv2.waitKey(0)
cv2.destroyAllWindows()