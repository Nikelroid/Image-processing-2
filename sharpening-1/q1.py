import numpy as np
import cv2


def clipping(img):
    return img.clip(0, 255)


def show_fourier(img, n):
    return n * np.log(np.abs(img) + 1)


def normalize(img):
    return (img - img.min()) * (255 / (img.max() - img.min()))


def uv_main_transform(img):
    reduced_uv_fourier_image = np.zeros_like(img)
    w = img.shape[1]
    h = img.shape[0]
    for i in range(0, h):
        for j in range(0, w):
            reduced_uv_fourier_image[i, j] = (((i - (h / 2)) ** 2) + ((j - (w / 2)) ** 2)) * img[i, j]
    return reduced_uv_fourier_image * (4 * (np.pi ** 2))


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


def get_gaussian_kernel(l, sig):
    limit = int(l / 2)
    raw = np.zeros((l, l))
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            raw[i + limit, j + limit] = np.exp(-((i ** 2) + (j ** 2)) / (2 * (sig ** 2)))
    gaussian_matrix = np.divide(raw, 2 * np.pi * (sig ** 2))
    return gaussian_matrix / (np.sum(gaussian_matrix))


def get_laplacian_of_gaussian_kernel(l, sig):
    limit = int(l / 2)
    raw = np.zeros((l, l))
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            raw[i + limit, j + limit] = ((i ** 2) + (j ** 2) - (2 * (sig ** 2))) * np.exp(
                -((i ** 2) + (j ** 2)) / (2 * (sig ** 2)))
    gaussian_matrix = np.divide(raw, 2 * np.pi * (sig ** 6))
    return gaussian_matrix * -1


def gaussian_sharpening():
    gauss_kernel = get_gaussian_kernel(5, 2)
    cv2.imwrite('res01.jpg', normalize(gauss_kernel))

    gauss_convolved = cv2.filter2D(image, -1, gauss_kernel)
    cv2.imwrite('res02.jpg', gauss_convolved)

    gauss_unsharp_mask = np.subtract(image_int, gauss_convolved)
    cv2.imwrite('res03.jpg', np.array(normalize(gauss_unsharp_mask), dtype='uint8'))

    alpha = 5
    gauss_final = clipping(np.add(image_int, alpha * gauss_unsharp_mask))
    cv2.imwrite('res04.jpg', np.array(gauss_final, dtype='uint8'))


def laplacian_sharpening():
    kernel_size = 13
    laplacian_kernel = get_laplacian_of_gaussian_kernel(kernel_size, 1)

    show_laplace1 = laplacian_kernel.copy()
    show_laplace1[show_laplace1 < 0] = 0
    show_laplace1 = normalize(show_laplace1)

    show_laplace2 = laplacian_kernel.copy()
    show_laplace2 *= -1
    show_laplace2[show_laplace2 < 0] = 0
    show_laplace2 = normalize(show_laplace2)

    final_laplacian_filter = cv2.merge((show_laplace1, np.zeros((kernel_size, kernel_size)), show_laplace2))
    cv2.imwrite('res05.png', final_laplacian_filter)

    laplacian_unsharp_mask = cv2.filter2D(image_int, -1, laplacian_kernel)

    cv2.imwrite('res06.jpg', normalize(laplacian_unsharp_mask))

    kappa = 6
    laplacian_final = clipping(np.add(image_int, kappa * np.array(laplacian_unsharp_mask, dtype='int')))
    cv2.imwrite('res07.jpg', np.array(laplacian_final, dtype='uint8'))


def fourier_highpass_sharpening():
    fourier_image = fourier_magnitude(image_int)
    cv2.imwrite('res08.jpg', np.array(show_fourier(fourier_image, 20), dtype='uint8'))

    radius = 135
    kernel_size = 205
    sigma = 85
    mask = np.zeros_like(image_int)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1)[0]
    mask = 255 - mask
    gauss_kernel = get_gaussian_kernel(kernel_size, sigma)
    highpass_image = cv2.filter2D(mask, -1, gauss_kernel)
    cv2.imwrite('res09.jpg', np.array(highpass_image, dtype='uint8'))

    kappa = 5
    convolve_mask = highpass_image / 255
    Fourier_unsharped_mask_image = np.multiply(np.add(1, np.multiply(kappa, convolve_mask)), fourier_image)
    cv2.imwrite('res10.jpg', np.array(show_fourier(Fourier_unsharped_mask_image, 20), dtype='uint8'))

    final_fourier_highpass_filter = inverse_fourier_magnitude(Fourier_unsharped_mask_image)
    cv2.imwrite('res11.jpg', np.array(final_fourier_highpass_filter, dtype='uint8'))


def fourier_uv_sharpening():
    fourier_image = fourier_magnitude(image_int)
    reduced_uv_fourier_image = uv_main_transform(fourier_image)
    cv2.imwrite('res12.jpg', np.array(show_fourier(reduced_uv_fourier_image, 20), dtype='uint8'))

    inverted_fourier_uv = non_normalized_nverse_fourier_magnitude(reduced_uv_fourier_image).real
    cv2.imwrite('res13.jpg', np.array(normalize(inverted_fourier_uv), dtype='uint8'))

    kappa = 10**-6 * 2
    fourier_uv_final = clipping(np.add(image_int, np.multiply(kappa, inverted_fourier_uv)))
    cv2.imwrite('res14.jpg', np.array(fourier_uv_final, dtype='uint8'))


image = cv2.imread("flowers.blur.png", 1)
image_int = np.array(image, dtype='int')
gaussian_sharpening()
laplacian_sharpening()
fourier_highpass_sharpening()
fourier_uv_sharpening()

cv2.waitKey(0)
cv2.destroyAllWindows()
