import cv2
import matplotlib.pyplot as plt
import numpy as np


def normalize_image(im, val_range, new_max=256):
    vmin, vmax = val_range
    return new_max * (im - vmin) / (vmin - vmax)

def blur_im(img, kernel_size=3):
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blur


def dilate_image(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size))
    kernel[0, 0], kernel[0, kernel_size - 1], kernel[kernel_size - 1, 0], kernel[
        kernel_size - 1, kernel_size - 1] = 0, 0, 0, 0
    kernel = kernel.astype(np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)
    return dilated


def threshold_image(img, threshold_value):
    ret, thresh = cv2.threshold(img, threshold_value, 256, cv2.THRESH_BINARY)
    return thresh


def preprocess_image(img, value_range, threshold_value, kernel_size=3):
    # todo: generalize to N functions in order, multiple kernels
    # img = normalize_image(img, value_range)
    return threshold_image(dilate_image(blur_im(img, kernel_size), kernel_size), threshold_value)


def get_contours(img, min_area=0):
    img = img.astype(np.uint8)
    contours, heir = cv2.findContours(image=img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contours = [con for con in contours if cv2.contourArea(con) > min_area]
    return contours


def draw_contours(img, contours, thick=3):
    #     todo: more space to play with the drawing
    # im_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    im_rgb = img
    cv2.drawContours(image=im_rgb, contours=contours, contourIdx=-1, color=(256, 256, 256), thickness=thick)
    return im_rgb


def get_contour_patch(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    return img[y:y + h, x:x + w]

