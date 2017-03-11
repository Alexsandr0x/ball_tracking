import cv2
import json
import numpy as np

file_directory = "config.json"
try:
    config_ = json.loads(open(file_directory).read())
except Exception as error:
    raise error

colors = config_['colors']['ball_jpg']
lower_color = np.array(colors[0])
upper_color = np.array(colors[1])

image_dir = config_["image_path"]

picture = cv2.imread(image_dir)

picture_no_noise = cv2.fastNlMeansDenoisingColored(picture, None, 15, 10, 7, 21)

picture_hsv = cv2.cvtColor(picture_no_noise, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(picture_hsv, lower_color, upper_color)

im_orange = picture_no_noise.copy()
im_orange[mask == 0] = 0

contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

if len(contours) > 0:
    c = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)

    if M["m00"] != 0:
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(picture_no_noise, center, 10, (0, 255, 0), -1)

cv2.imwrite('out/frame.png', picture_no_noise)
