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

fps = cap.get(cv2.CAP_PROP_FPS)

colors = config_['colors']['ball']
has_alien = config_.get('alien')
lower_color = np.array(colors[0])
upper_color = np.array(colors[1])
counter = 0
speed = None
dx = None
dy = None
center = None

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
    mask = cv2.inRange(frame, lower_color, upper_color)

    im_orange = frame.copy()
    im_orange[mask == 0] = 0

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

            set_x = [points[p_iter][0] for p_iter in range(0, 10)]
            set_y = [points[p_iter][1] for p_iter in range(0, 10)]

            dx = points[-10][0] - points[i][0]
            dy = points[-10][1] - points[i][1]

            z = np.polyfit(set_x, set_y, 1)

            dir_x = -1 if np.sign(dx) == 1 else 1

            cv2.line(frame, center, get_points(z[0], z[1], 1000 * dir_x), (255, 0, 0))

            break

    if len(points) > fps:
        speed = (np.array(points[0]) - np.array(points[1]))

    cv2.putText(frame, "dx: {}, dy: {}".format(dx, dy),
                (0, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

    cv2.putText(frame, "fps: {}".format(fps),
                (0, 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

    if center and has_alien:
        s_img = cv2.imread("others/ufo.png")
        s_img = cv2.resize(s_img, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_CUBIC)
        try:
            offset_x = center[1]
            offset_y = center[0]
            frame[offset_x:offset_x + s_img.shape[0], offset_y:offset_y + s_img.shape[1]] = s_img
        except:
            pass

    cv2.imshow('frame', frame)
    counter += 1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
