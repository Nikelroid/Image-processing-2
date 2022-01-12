import math
import cv2
import numpy as np


def show_image(mapped_image):
    mapped_image = np.array(mapped_image, 'uint8')
    cv2.imshow('mapped_image', mapped_image)
    cv2.imwrite('res18.jpg', mapped_image)


def find_value(H, x_y_prime, rgb, main_image):
    multiply = np.matmul(H, x_y_prime)
    mapped_coords = (multiply[0] / multiply[2], multiply[1] / multiply[2])
    integer_x = (int(mapped_coords[0]), int(mapped_coords[0] + 1))
    integer_y = (int(mapped_coords[1]), int(mapped_coords[1] + 1))
    S = ((mapped_coords[0] - integer_x[0]) * (mapped_coords[1] - integer_y[0]),
         ((integer_x[1] - mapped_coords[0]) * (mapped_coords[1] - integer_y[0])),
         ((integer_x[1] - mapped_coords[0]) * (integer_y[1] - mapped_coords[1])),
         ((mapped_coords[0] - integer_x[0]) * (integer_y[1] - mapped_coords[1])))
    value = int((S[0] * main_image[integer_y[1], integer_x[1], rgb]) + \
                (S[1] * main_image[integer_y[1], integer_x[0], rgb]) + \
                (S[2] * main_image[integer_y[0], integer_x[0], rgb]) + \
                (S[3] * main_image[integer_y[0], integer_x[1], rgb]))
    return value


def mapping(H, orginal_image):
    percent = 0
    mapped = np.zeros([h, w, 3], dtype='int')
    for rgb in range(3):
        for j in range(h):
            for i in range(w):
                x = [i, j, 1]
                mapped[j, i, rgb] = int(find_value(H, x, rgb, orginal_image))
            if percent != int(((j / h) * (1 / 3)) * 100 + (33 * rgb)):
                percent = int(((j / h) * (1 / 3)) * 100 + (33 * rgb))
                print(percent, '%')
    show_image(mapped)
    return 1


def find_w(coords_x, coords_y):
    w1 = math.sqrt(((coords_x[0] - coords_x[3]) ** 2) + ((coords_y[0] - coords_y[3]) ** 2))
    w2 = math.sqrt(((coords_x[1] - coords_x[2]) ** 2) + ((coords_y[1] - coords_y[2]) ** 2))
    return int((w1 + w2) / 2)


def find_h(coords_x, coords_y):
    h1 = math.sqrt(((coords_x[0] - coords_x[1]) ** 2) + ((coords_y[0] - coords_y[1]) ** 2))
    h2 = math.sqrt(((coords_x[2] - coords_x[3]) ** 2) + ((coords_y[2] - coords_y[3]) ** 2))
    return int((h1 + h2) / 2)


def make_matrix(coords_x, coords_y):
    global w, h
    w = find_w(coords_x, coords_y) * 2
    h = find_h(coords_x, coords_y) * 2
    A = np.array([
        [coords_x[0], coords_y[0], 1, 0, 0, 0, -0 * coords_x[0], -0 * coords_y[0]],
        [0, 0, 0, coords_x[0], coords_y[0], 1, -0 * coords_x[0], -0 * coords_y[0]],
        [coords_x[1], coords_y[1], 1, 0, 0, 0, -0 * coords_x[1], -0 * coords_y[1]],
        [0, 0, 0, coords_x[1], coords_y[1], 1, -h * coords_x[1], -h * coords_y[1]],
        [coords_x[2], coords_y[2], 1, 0, 0, 0, -w * coords_x[2], -w * coords_y[2]],
        [0, 0, 0, coords_x[2], coords_y[2], 1, -h * coords_x[2], -h * coords_y[2]],
        [coords_x[3], coords_y[3], 1, 0, 0, 0, -w * coords_x[3], -w * coords_y[3]],
        [0, 0, 0, coords_x[3], coords_y[3], 1, -0 * coords_x[3], -0 * coords_y[3]],
    ], dtype=np.float64)
    b = np.array([0, 0, 0, h, w, h, w, 0], dtype=np.float64)
    H_hat = np.linalg.solve(A, b)
    H = [
        [H_hat[0], H_hat[1], H_hat[2]],
        [H_hat[3], H_hat[4], H_hat[5]],
        [H_hat[6], H_hat[7], 1],
    ]
    return np.linalg.inv(H)


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        Xcoords.append(x)
        Ycoords.append(y)
        if len(Xcoords) > 0:
            cv2.line(image, (Xcoords[len(Xcoords) - 1], Ycoords[len(Ycoords) - 1]),
                     (Xcoords[len(Xcoords) - 2], Ycoords[len(Ycoords) - 2]), (255, 0, 0), 1)
        if len(Xcoords) == 4:
            cv2.line(image, (Xcoords[len(Xcoords) - 1], Ycoords[len(Ycoords) - 1]),
                     (Xcoords[0], Ycoords[0]), (255, 0, 0), 1)
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow('image', image)
            k = cv2.waitKey(20) & 0xFF
            H = make_matrix(np.multiply(Xcoords, 2), np.multiply(Ycoords, 2))
            mapping(H, org_image)
            print('100 %')
            print('DONE!')
            Xcoords.clear()
            Ycoords.clear()
        else:
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        mouseX, mouseY = x, y

Xcoords = []
Ycoords = []
image = cv2.imread("books.jpg", 1)
org_image = image.copy()
image = resize(image, 0.5)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while (1):
    cv2.imshow('image', image)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print(mouseX, mouseY)

cv2.waitKey(0)
cv2.destroyAllWindows()
