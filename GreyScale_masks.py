import numpy as np
import cv2
height, width = 640, 427

box = [[1, 93, 180, 110, 172],
       [2, 93, 124, 277, 386],
       [3, 193, 232, 155, 214]]
mask = np.zeros((height, width), dtype = np.int8)
for class_id, x_min, x_max, y_min, y_max in box:
    mask[x_min:x_max, y_min:y_max]=class_id
cv2.imwrite("image.png", mask)
cv2.imshow("image", mask*100)
cv2.waitKey(0)