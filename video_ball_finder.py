# coding=utf-8

import cv2
import json

import numpy as np

from collections import deque
from os import listdir
from os.path import isfile, join

file_directory = "config.json"
try:
    config_ = json.loads(open(file_directory).read())
except Exception as error:
    raise error

video_path = config_['video_path']

cap = cv2.VideoCapture(video_path)

colors = config_['colors']['ball']
feature = config_.get('feature')
lower_color = np.array(colors[0])
upper_color = np.array(colors[1])
center = [0, 0]

if feature:
    if feature == 'ronaldinho':
        path_ = 'others/ronaldinho/'
        feat_image = [
            path_ + f for f in listdir(path_) if (
                isfile(join(path_, f)) and f.endswith(".jpg"))
        ]
        feat_image = sorted(feat_image)
    elif feature == 'alien':
        feat_image = "others/ufo.png"
    else:
        print 'feature not found try these guys[alien, ronaldinho]'
        feature = None

point_buffer = 64
points = deque(maxlen=point_buffer)

ronaldinho_frame_counter = 0
ronaldinho_frame_time_counter = 0
ronaldinho_frame_time_map = json.loads(open('others/ronaldinho/ronaldinho.json').read())
writer = None


def get_points(a, b, x_):
    y_ = x_ * a + b
    return int(x_), int(y_)


def add_alien(center, frame):
    s_img = cv2.imread(feat_image)
    s_img = cv2.resize(s_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    try:
        offset_x = center[1]
        offset_y = center[0]
        # GAMBI para fazer a camada "alfa" funcionar para o png.
        for x_ in range(0, s_img.shape[0]):
            for y_ in range(0, s_img.shape[1]):
                if not (s_img[x_, y_] == [255, 255, 255]).all():
                    frame[
                        offset_x - s_img.shape[0] / 2 + x_,
                        offset_y - s_img.shape[1] / 2 + y_
                    ] = s_img[x_, y_]
    except Exception:
        return frame

    return frame


def add_ronaldinho(center, frame):
    global ronaldinho_frame_counter
    global ronaldinho_frame_time_counter

    s_img = cv2.imread(feat_image[ronaldinho_frame_counter % len(feat_image)])
    s_img = cv2.resize(s_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    try:
        offset_x = center[1]
        offset_y = center[0]
        # GAMBI para fazer a camada "alfa" funcionar para o png.
        for x_ in range(0, s_img.shape[0]):
            for y_ in range(0, s_img.shape[1]):
                if not (s_img[x_, y_] == [254, 0, 0]).all():
                    frame[
                        offset_x - s_img.shape[0] / 2 + x_,
                        offset_y - s_img.shape[1] / 2 + y_
                    ] = s_img[x_, y_]
    except Exception:
        return frame

    ronaldinho_frame_counter += 1
    ronaldinho_frame_time_counter += 1
    return frame


while True:

    # Take each frame
    _, frame = cap.read()
    # frame = imutils.resize(frame, width=600)
    if frame is None:
        break

    if writer is None:
        # store the image dimensions, initialzie the video writer,
        # and construct the zeros array
        (h, w) = frame.shape[:2]
        four_cc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter('out/video.avi', four_cc, 40, (w, h), True)
        zeros = np.zeros((h, w), dtype="uint8")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

    cv2.putText(frame, "x: {}, y: {}".format(center[0], center[1]),
                (0, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

    if center and feature:
        if feature == 'alien':
            frame = add_alien(center, frame)
        elif feature == 'ronaldinho':
            frame = add_ronaldinho(center, frame)

    cv2.imshow('frame', frame)
    writer.write(frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
writer.release()
