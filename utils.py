import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

REDU = 8


def rgbh(xs, mask):
    def normhist(x): return x / np.sum(x)

    def h(rgb):
        return cv2.calcHist([rgb],
                            [0, 1, 2],
                            mask,
                            [256 // REDU, 256 // REDU, 256 // REDU],
                            [0, 256] + [0, 256] + [0, 256])

    return normhist(sum(map(h, xs)))


def smooth(s, x):
    return gaussian_filter(x, s, mode='constant')
