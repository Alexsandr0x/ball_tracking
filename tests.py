from collections import deque

import cv2
import imutils
import json
import numpy as np

file_directory = "config.json"
try:
    config_ = json.loads(open(file_directory).read())
except Exception as error:
    raise error

video_path = config_['video_path']
cap = cv2.VideoCapture(video_path)

colors = config_['colors']['ball']
lower_blue = np.array(colors[0])
upper_blue = np.array(colors[1])
counter = 0
dx = None
dy = None

point_buffer = 64
points = deque(maxlen=point_buffer)


def get_points(a, b, x_):
    y_ = x_ * a + b
    return int(x_), int(y_)

while True:

    # Take each frame
    _, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(frame, lower_blue, upper_blue)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            points.appendleft(center)
            cv2.circle(frame, center, 10, (0, 255, 0), -1)

    for i in np.arange(1, len(points)):
        # if either of the tracked points are None, ignore
        # them
        if points[i - 1] is None or points[i] is None:
            continue

        # check to see if enough points have been accumulated in
        # the buffer
        if len(points) >= 10 and i == 1 and points[-10] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            dx = points[-10][0] - points[i][0]
            dy = points[-10][1] - points[i][1]

            set_x = [points[p_iter][0] for p_iter in range(0, 10)]
            set_y = [points[p_iter][1] for p_iter in range(0, 10)]
            z = np.polyfit(set_x, set_y, 1)

            cv2.line(frame, get_points(z[0], z[1], 0), get_points(z[0], z[1], 1000), (255, 0, 0))

            break

    cv2.putText(frame, "dx: {}, dy: {}".format(dx, dy),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 0), 1)

    cv2.imshow('frame', frame)
    counter += 1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
