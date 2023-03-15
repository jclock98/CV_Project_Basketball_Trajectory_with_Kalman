from typing import Dict

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from const import CUT_FACTOR, REDU


def get_bbox_coords(results: Dict) -> (int, int, int, int):
    bbox = results[0].boxes.xyxy.numpy()[0]
    x_t_l, y_t_l, x_b_r, y_b_r = list(map(lambda x: int(x), bbox))
    x_b_r -= CUT_FACTOR
    y_b_r -= CUT_FACTOR
    x_t_l += CUT_FACTOR
    y_t_l += CUT_FACTOR
    return x_t_l, y_t_l, x_b_r, y_b_r


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
