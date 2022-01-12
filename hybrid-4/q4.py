from pprint import pprint

import cv2
import numpy as np


def get_gaussian_kernel(l, sig):
    limit = int(l / 2)
    raw = np.zeros((l, l))
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            raw[i + limit, j + limit] = np.exp(-((i ** 2) + (j ** 2)) / (2 * (sig ** 2)))
    gaussian_matrix = np.divide(raw, 2 * np.pi * (sig ** 2))
    return gaussian_matrix / (np.sum(gaussian_matrix))


def highpass(sigma, image_int):
    h = image_int.shape[0]
    w = image_int.shape[1]
    gauss_kernel = get_gaussian_kernel(np.max(image_int.shape)+1, sigma)
    result = normalize_binary(gauss_kernel)
    result = 1 - result
    dif = int(np.abs(h - w)/2)
    if h > w:
        result = result[:h, dif:dif + w]
    elif w < h:
        result = result[dif:dif + h, :w]
    else:
        result = result[:h-1,:w-1]
    cv2.imwrite('res25-highpass-r.jpg', normalize(result))

    return cv2.merge((result,result,result))


def lowpass(sigma, image_int):
    h = image_int.shape[0]
    w = image_int.shape[1]
    gauss_kernel = get_gaussian_kernel(np.max(image_int.shape) + 1, sigma)
    result = normalize_binary(gauss_kernel)
    dif = int(np.abs(h - w) / 2)
    if h > w:
        result = result[:h, dif:dif + w]
    elif w < h:
        result = result[dif:dif + h, :w]
    else:
        result = result[:h - 1, :w - 1]
    cv2.imwrite('res26-lowpass-s.jpg', normalize(result))
    return cv2.merge((result, result, result))


def show_fourier(img, n):
    return n * np.log(np.abs(img) + 1)


def normalize(img):
    return (img - img.min()) * (255 / (img.max() - img.min()))


def normalize_binary(img):
    return (img - img.min()) * (1 / (img.max() - img.min()))


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def fourier_magnitude(img):
    f = np.fft.fft2(img, axes=(0, 1))
    fshift = np.fft.fftshift(f)
    return fshift


def inverse_fourier_magnitude(img):
    f_ishift = np.fft.ifftshift(img)
    img_back = np.fft.ifft2(f_ishift, axes=(0, 1))
    return np.abs(img_back).clip(0, 255).astype(np.int32)


def non_normalized_nverse_fourier_magnitude(img):
    f_ishift = np.fft.ifftshift(img)
    img_back = np.fft.ifft2(f_ishift, axes=(0, 1))
    return img_back


def zero(h, w):
    return np.zeros((h, w, 3), dtype='uint8')


def match_dimes(image1, image2):
    v_dif = int(abs(image1.shape[1] - image2.shape[1]) / 2)
    if v_dif != 0:
        if image1.shape[1] > image2.shape[1]:
            image2 = np.hstack((zero(image2.shape[0], v_dif), image2, zero(image2.shape[0], v_dif)))
        else:
            image1 = np.hstack((zero(image1.shape[0], v_dif), image1, zero(image1.shape[0], v_dif)))

    h_dif = int(abs(image1.shape[0] - image2.shape[0]) / 2)
    if h_dif != 0:
        if image1.shape[0] > image2.shape[0]:
            image2 = np.vstack((zero(h_dif, image2.shape[1]), image2, zero(h_dif, image2.shape[1])))
        else:
            image1 = np.vstack((zero(h_dif, image1.shape[1]), image1, zero(h_dif, image1.shape[1])))

    return image1, image2


def make_dimes_even(image1, image2):
    image1 = image1[:(image1.shape[0] // 2) * 2, :(image1.shape[1] // 2) * 2]
    image2 = image2[:(image2.shape[0] // 2) * 2, :(image2.shape[1] // 2) * 2]
    return image1, image2


def expand(image1, search_fraction):
    main_dimes = image1.shape[:2]
    w = int(image1.shape[1] / search_fraction)
    h = int(image1.shape[0] / search_fraction)
    image1 = np.hstack((zero(image1.shape[0], w), image1, zero(image1.shape[0], w)))
    image1 = np.vstack((zero(h, image1.shape[1]), image1, zero(h, image1.shape[1])))
    return image1, int(main_dimes[1] + (w * 2)), int(main_dimes[0] + (w * 2))


image1 = cv2.imread("raw1.jpg", 1)
image2 = cv2.imread("raw2.jpg", 1)

cv2.imwrite('res19-near.jpg',image1)
cv2.imwrite('res20-far.jpg',image2)

image1, image2 = make_dimes_even(image1, image2)
image1, image2 = match_dimes(image1, image2)
h_old, w_old = image1.shape[:2]
base, w_new, h_new = expand(image1, 10)

base_int = np.array(resize(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY), 0.5), dtype='int')
image2_int = np.array(resize(cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY), 0.5), dtype='int')

h_old, w_old, w_new, h_new = h_old // 2, w_old // 2, w_new // 2, h_new // 2

maximum = 0
matched = (0, 0)
kappa = 500000
print(w_new, w_old, h_new, h_old)
for i in range(w_new - w_old):
    for j in range(h_new - h_old):
        cheking_base = base_int[j: j + h_old, i: i + w_old]
        match_number = (j * kappa) + np.sum(np.multiply(cheking_base, image2_int))
        if match_number > maximum:
            maximum = match_number
            matched = (j, i)

matched = np.multiply(matched, 2)

image1 = base[matched[0]:matched[0] + h_old * 2, matched[1]:matched[1] + w_old * 2]
image1, image2 = make_dimes_even(image1, image2)
image1, image2 = match_dimes(image1, image2)

head_point_i, head_point_j = 0, 0

imagecheck = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

for i in range(imagecheck.shape[0]):
    if np.average(imagecheck[i, :]) > 3:
        head_point_i = i
        head_array = []
        for j in range(imagecheck.shape[1]):
            if imagecheck[i * 2, j] > 30:
                head_array.append(j)
        head_array = np.array(head_array)
        head_point_j = int(np.average(head_array))
        break

image_gray_2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

maximum = 0
matched_scale = 1
center = (head_point_j, head_point_i)
srcTri = np.array([[head_point_j, head_point_i], [head_point_j + 100, head_point_i + 100],
                   [head_point_j + 100, head_point_i - 100]]).astype(np.float32)
scale_coefficant = 0.25
for i in range(1, 1000):
    scale_y = i / 100
    scale_x = scale_y / scale_coefficant
    dstTri = np.array([[head_point_j, head_point_i], [head_point_j + 100 + scale_y, head_point_i + 100 + scale_x],
                       [head_point_j + 100 + scale_y, head_point_i - 100 - scale_x]]).astype(np.float32)

    scale_mat = cv2.getAffineTransform(srcTri, dstTri)
    warp_rotate_dst = cv2.warpAffine(imagecheck, scale_mat, (imagecheck.shape[1], imagecheck.shape[0]))[:h_old * 2,
                      :w_old * 2]
    match_number = (100000 / i) + np.sum(np.multiply(warp_rotate_dst.astype('int32'), image_gray_2.astype('int32')))
    if match_number > maximum:
        maximum = match_number
        matched_scale = (scale_y, scale_x)

print(matched_scale)
dstTri = np.array(
    [[head_point_j, head_point_i], [head_point_j + 100 + (matched_scale[0]), head_point_i + 100 + matched_scale[1]],
     [head_point_j + 100 + matched_scale[0], head_point_i - 100 - matched_scale[1]]]).astype(np.float32)
scale_mat = cv2.getAffineTransform(srcTri, dstTri)
image1 = cv2.warpAffine(image1, scale_mat, (image1.shape[1], image1.shape[0]))

cv2.imwrite('res21-near.jpg', np.array(image1, dtype='uint8'))
cv2.imwrite('res22-far.jpg', np.array(image2, dtype='uint8'))

fourier_image1 = fourier_magnitude(image1.astype('int'))
fourier_image2 = fourier_magnitude(image2.astype('int'))

cv2.imwrite('res24-dft-far.jpg', np.array(show_fourier(fourier_image1, 20), dtype='uint8'))
cv2.imwrite('res23-dft-near.jpg', np.array(show_fourier(fourier_image2, 20), dtype='uint8'))

highpass_filter = highpass(105, image1.astype('int'))
highpass_image1 = np.multiply(fourier_image1, highpass_filter)

lowpass_filter = lowpass(10, image2.astype('int'))
lowpass_image2 = np.multiply(fourier_image2, lowpass_filter)

a =1.8
fourier_sum = np.multiply(highpass_image1, a) + np.multiply(lowpass_image2, (a - 2.6))

cv2.imwrite('res27-highpassed.jpg', np.array(show_fourier(highpass_image1, 20), dtype='uint8'))
cv2.imwrite('res28-lowpassed.jpg', np.array(show_fourier(lowpass_image2, 20), dtype='uint8'))
cv2.imwrite('res29-hybrid.jpg', np.array(show_fourier(fourier_sum, 20), dtype='uint8'))

result = inverse_fourier_magnitude(fourier_sum)

cv2.imwrite('res30-hybrid-near.jpg', result.astype('uint8'))
cv2.imwrite('res31-hybrid-far.jpg', resize(result.astype('uint8'),0.25))

convolved = cv2.addWeighted(image1, 0.5, image2, 0.5, 0.0)
# cv2.imshow('3', convolved)

cv2.waitKey(0)
cv2.destroyAllWindows()
