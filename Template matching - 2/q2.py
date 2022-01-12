import argparse
import math
import statistics
from pprint import pprint

import cv2
import numpy as np


def cut_template(tmp):
    temp_canny = cv2.Canny(tmp, 5.0, 300.0)
    w = tmp.shape[1]
    h = tmp.shape[0]

    end_x = w
    start_x = 0
    end_y = h
    start_y = 0

    maximum = 0
    for i in range(int(w / 2), int(w - 4)):
        total = np.sum(temp_canny[:, i:i + 2])
        if total > maximum:
            maximum = total
            end_x = i + 12

    maximum = 0
    for i in range(-int(w / 2), -int(4)):
        i *= -1
        total = np.sum(temp_canny[:, i:i + 2])
        if total > maximum:
            maximum = total
            start_x = i - 12
    maximum = 0
    for i in range(int(h / 2), int(h - 4)):
        total = np.sum(temp_canny[i:i + 2, :])
        if total > maximum:
            maximum = total
            end_y = i

    maximum = 0
    for i in range(-int(h / 2), -int(4)):
        i *= -1
        total = np.sum(temp_canny[i:i + 2, :])
        if total > maximum:
            maximum = total
            start_y = i - 5

    print(h, w, start_y - end_y, start_x - end_x)
    return template[start_y:end_y, start_x:end_x]


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


print("[INFO] loading images...")
ship = cv2.imread("Greek-ship.jpg", 1)
image = cv2.cvtColor(ship, cv2.COLOR_RGB2GRAY)
template = cv2.imread("patch.png", 0)

rate = 0.25
image = resize(image, rate)

template = cut_template(template)

template = resize(template, rate)
image = np.array(image, dtype='int')
template = np.array(template, dtype='int')

(tH, tW) = template.shape[:2]
(iH, iW) = image.shape[:2]
minimum = 1000000000000
percent = 0
Xcords = []
Ycords = []
mean_t = np.array([[np.mean(template)] * tW] * tH, dtype='int')
half_w = int(tW / 2)
half_h = int(tH / 2)

for i in range(iH - tH):
    for j in range(iW - tW):
        cut_image = image[i:i + tH, j:j + tW]
        total = np.sum((template - [[np.mean(cut_image)] * tW] * tH) * (cut_image - mean_t))
        if total > 300000:
            Ycords.append(i)
            Xcords.append(j)

    if int(101 * i / (iH - tH)) != percent:
        percent = int(101 * i / (iH - tH))
        print(percent, '%')

linked_points_x = [[Xcords[0]]]
linked_points_y = [[Ycords[0]]]

for i in range(len(Xcords)):
    flag = True
    for x in range(len(linked_points_x)):
        link = np.array(linked_points_x[x])
        print(np.mean(link) - half_w, Xcords[i], np.mean(link) + half_w, flag)
        if np.mean(link) - half_w < Xcords[i] < np.mean(link) + half_w:
            flag = False
            linked_points_x[x].append(Xcords[i])
            linked_points_y[x].append(Ycords[i])
            break
    if flag:
        linked_points_x.append([Xcords[i]])
        linked_points_y.append([Ycords[i]])

new_cords_x = [int(np.mean(linked_points_x[0]))]
new_cords_y = [int(np.mean(linked_points_y[0]))]

for x in range(1,len(linked_points_x)):
    new_cords_x.append(int(np.mean(np.array(linked_points_x[x]))))
    new_cords_y.append(int(np.mean(np.array(linked_points_y[x]))))



for rec_y, rec_x in zip(new_cords_y, new_cords_x):
    cv2.rectangle(ship, (rec_x*4, rec_y*4), (rec_x*4 + tW*4, rec_y*4 + tH*4), (255, 0, 0), 4)

cv2.imwrite('res29.jpg', ship)
cv2.imshow('res', resize(ship,0.25))

cv2.waitKey(0)
cv2.destroyAllWindows()
