import numpy as np
import cv2
height, width = 440, 627
image_path = "C:/Users/namra/OneDrive/Pictures/Saved Pictures/Computer_science_and_engineering.jpg"
original = cv2.imread(image_path)
box = [[1, 93, 180, 110, 172],
       [2, 93, 124, 277, 386],
       [3, 193, 232, 155, 214]]
#mask = np.zeros((height, width), dtype = np.int8)
text_labels = {
        1: "AI&ML",
        2: "System",
        3: "Data"
        }
for class_id, x_min, x_max, y_min, y_max in box:
    original[x_min:x_max, y_min:y_max]=class_id
    text = text_labels.get(class_id, "Unknown")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White color
    font_thickness = 1
    text_position = (y_min + 5, x_min + 25)

        # Use cv2.putText() to write the text on the image
    cv2.putText(original, text, text_position, font, font_scale, font_color, font_thickness)

cv2.imwrite("image.png", original)
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", original)
cv2.waitKey(0)