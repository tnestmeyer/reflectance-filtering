# MIT License
#
# Copyright (c) 2017 Thomas Nestmeyer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Some image utils generally needed.
"""
from __future__ import division, print_function

import cv2
import numpy as np


def srgb_to_rgb(srgb):
    """Taken from bell2014: sRGB -> RGB."""
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def rgb_to_srgb(rgb):
    """Taken from bell2014: RGB -> sRGB."""
    ret = np.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = np.power(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret


def imread(filename):
    """Read image with OpenCV and check state afterwards."""
    img = cv2.imread(filename)
    if img is None:
        raise Exception("Input image not readable: {}".format(filename))
    return img


def imwrite(filename, image, sRGB=False):
    """Write image with OpenCV and check state afterwards."""
    # if the image is not in uint8 format, normalize it first
    if image.dtype != np.uint8:
        image = normalize(image)
        if sRGB:
            image = rgb_to_srgb(image)
        # make range from 0-1 float into 0-255 uint8
        image = (image * 255).astype(np.uint8)
    # now write the file and check if succeeded
    success = cv2.imwrite(filename, image)
    if not success:
        msg = ("Not able to write {}, does the folder exist?".format(filename))
        raise Exception(msg)


def colorize(intensity, image, eps=1e-3):
    """Reconstruct color image from reflectance intensity and input image."""
    norm_input = np.mean(image, axis=2)
    shading = norm_input / intensity
    reflectance = image / np.maximum(shading, eps)[:, :, np.newaxis]
    return reflectance, shading


def normalize(img):
    """Normalize image to a range of 0-1."""
    img = img.copy()
    if np.max(img) > 1:
        # img /= np.max(img)
        # # ignore some overly high pixel values in 0.1% of the image:
        img /= np.percentile(img, 99.9, interpolation='lower')
        img = np.clip(img, 0, 1)
    return img
