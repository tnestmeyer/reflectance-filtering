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
Contains the RecoverReflectanceShadingLayer.

Contains the RecoverReflectanceShadingLayer that recovers RGB reflectance
and shading from a (possibly scalar) estimation of one of the two and an input
image. See the description of the class.
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
# import timeit
from skimage.color import rgb2lab, lab2rgb
import cv2

try:
    import caffe
except ImportError:
    sys.path.insert(0, os.path.join(os.path.expanduser('~'),
                                    'Repositories',
                                    'malabar',
                                    'python'))
    import caffe


# np.finfo(np.float32).eps = 1.1920929e-07
# np.finfo(np.float).eps = 2.2204460492503131e-16
EPS = np.finfo(np.float32).eps

# RGB2LIGHTNESS = [0.299, 0.587, 0.114]  # not used right now


class RecoverReflectanceShadingLayer(caffe.Layer):
    """
    Recover RGB reflectance and shading from estimation and input image.

    Depending on interpretation, the input can be either an estimation of
    reflectance, or shading. Further, it can be a one channel estimation, if
    we limit ourselves to white light, or a three channel estimation.
    """

    def setup(self, bottom, top):
        """Check that layer is correctly set up by checking input pair."""
        if len(bottom) != 2:
            raise Exception("Expected inputs:\n"
                            "0) Estimation of reflectance or shading,\n"
                            "1) input images.")

        def print_shapes():
            """Print the input shapes."""
            print("bottom[0].data.shape:", bottom[0].data.shape,
                  "\nbottom[1].data.shape:", bottom[1].data.shape)

        # check that inputs match
        if not (bottom[0].data.ndim == bottom[1].data.ndim == 4):
            print_shapes()
            raise Exception("Expecting 4D blobs on both inputs.")
        if bottom[0].data.shape[0] != bottom[1].data.shape[0]:
            print_shapes()
            raise Exception("Batch sizes of the inputs do not match!")
        if bottom[0].data.shape[2] != bottom[1].data.shape[2]:
            print_shapes()
            raise Exception("Heights of the input images do not match!")
        if bottom[0].data.shape[3] != bottom[1].data.shape[3]:
            print_shapes()
            raise Exception("Widths of the input images do not match!")

        self.eps = EPS

        # now set up which function to use (how to interpret input)
        params = self.param_str.split('-')
        mode = params[0]
        # print("Setting up RecoverReflectanceShadingLayer.",
        #       "mode is:", mode)
        possible_modes = ['sAbs', 'S', 'rAbs', 'R', 'RS',
                          'rRelNorm', 'rRelMean', 'rRelY', 'rRelMax',
                          'sRelNorm', 'sRelMean', 'sRelY', 'sRelMax',
                          'CIELAB', 'HLS', 'HSV',
                          ]
        if mode not in possible_modes:
            print("given first part (before '-') of param_str '", mode,
                  "' in RecoverReflectanceShadingLayer was not one of",
                  "the expected: " + str(possible_modes) + ".",
                  "rRelNorm is chosen as standard")
            mode = 'rRelMax'

        if mode == 'sAbs':
            self.f = self.interpret_input_as_shading_intensity_absolute
            if bottom[0].data.shape[1] != 1:
                raise Exception("The input (interpreted as scalar shading) "
                                "is supposed to have one channel")
        elif mode == 'sRelNorm':
            self.f = self.interpret_input_as_shading_intensity_relative
            if bottom[0].data.shape[1] != 1:
                raise Exception("The input (interpreted as scalar shading) "
                                "is supposed to have one channel")
            self.norm = _norm_L2norm
        elif mode == 'sRelMean':
            self.f = self.interpret_input_as_reflectance_intensity_relative
            if bottom[0].data.shape[1] != 1:
                raise Exception("The input (interpreted as scalar reflectance)"
                                " is supposed to have one channel")
            self.norm = _norm_Mean
        elif mode == 'sRelY':
            self.f = self.interpret_input_as_reflectance_intensity_relative
            if bottom[0].data.shape[1] != 1:
                raise Exception("The input (interpreted as scalar reflectance)"
                                " is supposed to have one channel")
            self.norm = _norm_Lightness
        elif mode == 'sRelMax':
            self.f = self.interpret_input_as_reflectance_intensity_relative
            if bottom[0].data.shape[1] != 1:
                raise Exception("The input (interpreted as scalar reflectance)"
                                " is supposed to have one channel")
            self.norm = _norm_Max
        elif mode == 'S':
            self.f = self.interpret_input_as_shading_RGB
            if bottom[0].data.shape[1] != 3:
                raise Exception("The input (interpreted as RGB shading) "
                                "is supposed to have three channels")
        elif mode == 'rAbs':
            self.f = self.interpret_input_as_reflectance_intensity_absolute
            if bottom[0].data.shape[1] != 1:
                raise Exception("The input (interpreted as scalar reflectance)"
                                " is supposed to have one channel")
        elif mode == 'rRelNorm':
            self.f = self.interpret_input_as_reflectance_intensity_relative
            if bottom[0].data.shape[1] != 1:
                raise Exception("The input (interpreted as scalar reflectance)"
                                " is supposed to have one channel")
            self.norm = _norm_L2norm
        elif mode == 'rRelMean':
            self.f = self.interpret_input_as_reflectance_intensity_relative
            if bottom[0].data.shape[1] != 1:
                raise Exception("The input (interpreted as scalar reflectance)"
                                " is supposed to have one channel")
            self.norm = _norm_Mean
        elif mode == 'rRelY':
            self.f = self.interpret_input_as_reflectance_intensity_relative
            if bottom[0].data.shape[1] != 1:
                raise Exception("The input (interpreted as scalar reflectance)"
                                " is supposed to have one channel")
            self.norm = _norm_Lightness
        elif mode == 'rRelMax':
            self.f = self.interpret_input_as_reflectance_intensity_relative
            if bottom[0].data.shape[1] != 1:
                raise Exception("The input (interpreted as scalar reflectance)"
                                " is supposed to have one channel")
            self.norm = _norm_Max
        elif mode == 'R':
            self.f = self.interpret_input_as_reflectance_RGB
            if bottom[0].data.shape[1] != 3:
                raise Exception("The input (interpreted as RGB reflectance)"
                                " is supposed to have three channels")
        elif mode == 'RS':
            self.f = self.interpret_input_as_concatenation_of_R_and_S
            if bottom[0].data.shape[1] != 6:
                raise Exception("The input (interpreted as RGB reflectance "
                                "concatenated with RGB shading) is supposed "
                                "to have six channels")

    def reshape(self, bottom, top):
        """Define dimensions of data."""
        # output:
        # 0) reflectance
        # 1) shading
        batch_size, channels, height, width = bottom[1].data.shape
        top[0].reshape(batch_size, channels, height, width)
        top[1].reshape(batch_size, channels, height, width)

    def forward(self, bottom, top):
        """Forward pass of the layer."""
        # start = timeit.default_timer()
        # print("RecoverReflectanceShadingLayer: min, max, mean R_i:",
        #       np.min(R_i), np.max(R_i), np.mean(R_i))
        self.f(bottom, top)
        # stop = timeit.default_timer()
        # print("RecoverReflectanceShadingLayer: ",
        #       "Time to recover reflectance and shading:",
        #       stop-start, "seconds.")

    def backward(self, top, propagate_down, bottom):
        """Backward pass of the layer."""
        # print("RecoverReflectanceShadingLayer: "
        #       "Computing backward to R_i (bottom[0]):", propagate_down[0])
        # print("RecoverReflectanceShadingLayer: "
        #       "Computing backward to images (bottom[1]):", propagate_down[1])

        if propagate_down[0]:
            diff_reflectance = top[0].diff * self.diff_reflectance
            diff_shading = top[1].diff * self.diff_shading
            # print(top[0].diff.shape, top[1].diff.shape,
            #       self.diff_reflectance.shape, self.diff_shading.shape)
            if bottom[0].data.shape[1] == 6:
                diff_input = np.concatenate((diff_reflectance, diff_shading),
                                            axis=1)
            elif bottom[0].data.shape[1] == 3:
                # diff_input = diff_reflectance
                # diff_input = diff_shading
                # diff_input = np.sqrt(diff_reflectance**2 + diff_shading**2)

                diff_input = diff_reflectance + diff_shading
            elif bottom[0].data.shape[1] == 1:
                # since every RGB component has same influence, we propagate
                # their sum
                diff_input = np.sum(np.concatenate((diff_reflectance,
                                                    diff_shading),
                                                   axis=1),
                                    axis=1, keepdims=True)
                # TODO: do we need to compute the correct weighting when using
                # the proper RGB weighting for Y?
            else:
                Exception("Num channels is expected to be one of [1, 3, 6].",
                          bottom[0].data.shape[1])
            bottom[0].diff[...] = diff_input

        if propagate_down[1]:
            print("RecoverReflectanceShadingLayer: "
                  "can't propagate down to images!")
            # raise Exception("There should be no diff towards images!")

    def interpret_input_as_reflectance_intensity_relative(self, bottom, top):
        """Interpret input as reflectance intensity."""
        # use real input
        # R_i = bottom[0].data
        # image = bottom[1].data

        # threshold with eps to avoid division by zero
        R_i = _threshold(bottom[0].data)
        # image = np.maximum(bottom[1].data, eps)
        image = bottom[1].data
        # probably better to threshold intensity, instead of image, see below

        # NOTE: diff of R_i should not change when changing intens, since
        # only dependent on input image, not estimation!
        # threshold
        image_intensity = _threshold(self.norm(image))

        # recover reflectance and shading by equation (5) in iiw:
        # first get the factor: Ri/intens
        factor = 1. / image_intensity
        # normalize image
        normalized_image = factor * image
        # get reflectance
        reflectance = R_i * normalized_image

        # now we want intens/Ri as the factor for shading
        Ri_inv = 1. / R_i
        factor = image_intensity * Ri_inv
        # now get shading
        shading = factor * np.ones_like(image)

        # assign the output:
        top[0].data[...] = reflectance
        top[1].data[...] = shading

        # save already computed values for backprop
        self.diff_reflectance = normalized_image
        self.diff_shading = -Ri_inv * shading

    def interpret_input_as_reflectance_intensity_absolute(self, bottom, top):
        """Interpret input as reflectance intensity."""
        # threshold with eps to avoid division by zero
        R_i = _threshold(bottom[0].data)
        image = bottom[1].data

        # get reflectance
        reflectance = R_i * image

        Ri_inv = 1. / R_i
        # now get shading
        shading = Ri_inv * np.ones_like(image)

        # assign the output:
        top[0].data[...] = reflectance
        top[1].data[...] = shading

        # save already computed values for backprop
        self.diff_reflectance = image
        self.diff_shading = -Ri_inv * shading

    def interpret_input_as_shading_intensity_relative(self, bottom, top):
        """Interpret input as shading intensity relative to image intensity."""
        # threshold with eps to avoid division by zero
        s = _threshold(bottom[0].data)
        image = bottom[1].data

        image_intensity = _threshold(self.norm(image))

        s_inv = 1. / s

        # get reflectance
        reflectance = image / image_intensity * s_inv

        # now get shading
        grayscale = image_intensity * np.ones_like(image)
        shading = grayscale * s

        # assign the output:
        top[0].data[...] = reflectance
        top[1].data[...] = shading

        # save already computed values for backprop
        self.diff_reflectance = -s_inv * reflectance
        self.diff_shading = grayscale

    def interpret_input_as_shading_intensity_absolute(self, bottom, top):
        """
        Interpret input as shading intensity.

        Interpret input as shading intensity independent of image intensity.
        """
        # threshold with eps to avoid division by zero
        s = _threshold(bottom[0].data)
        s_inv = 1. / s

        image = bottom[1].data

        reflectance = image * s_inv
        shading = s * np.ones_like(image)

        # assign the output:
        top[0].data[...] = reflectance
        top[1].data[...] = shading

        # save already computed values for backprop
        self.diff_reflectance = -s_inv * reflectance
        self.diff_shading = np.ones_like(image)

    def interpret_input_as_reflectance_RGB(self, bottom, top):
        """Interpret input as RGB reflectance layer."""
        # threshold with eps to avoid division by zero
        R = _threshold(bottom[0].data)
        I = bottom[1].data
        S = I / R

        # assign the output:
        top[0].data[...] = R
        top[1].data[...] = S

        # compute gradients (w.r.t input R) for backprop
        self.diff_reflectance = np.ones_like(R)
        self.diff_shading = -S / R

    def interpret_input_as_shading_RGB(self, bottom, top):
        """Interpret input as RGB shading layer."""
        # threshold with eps to avoid division by zero
        S = _threshold(bottom[0].data)
        I = bottom[1].data
        R = I / S

        # assign the output:
        top[0].data[...] = R
        top[1].data[...] = S

        # compute gradients (w.r.t input S) for backprop
        self.diff_reflectance = -R / S
        self.diff_shading = np.ones_like(S)

    def interpret_input_as_concatenation_of_R_and_S(self, bottom, top):
        """
        Interpret input as a concatenation of reflectance and shading.

        Interpret input as a concatenation of a RGB reflectance and a
        RGB shading layer.
        """
        R = bottom[0].data[:, 0:3, :, :]
        S = bottom[0].data[:, 3:6, :, :]

        # assign the output:
        top[0].data[...] = R
        top[1].data[...] = S

        # compute gradients (w.r.t input [R, S]) for backprop
        self.diff_reflectance = np.ones_like(R)
        self.diff_shading = np.ones_like(S)


def _norm_Mean(image):
    # normalize by 1/3 (r+g+b) as in Bell et al. 2014
    return np.sum(image, axis=1, keepdims=True) / 3


def _norm_L2norm(image):
    # normalize by 2-norm of color
    return np.linalg.norm(image, axis=1)[:, np.newaxis, :, :]


def _norm_Lightness(image):
    # normalize by using human lightness perception
    # the used version below is even a bit faster than with tensordot
    # return np.tensordot(image, RGB2LIGHTNESS, axes=(1, 0))
    return (0.299 * image[:, 0:1, :, :] +
            0.587 * image[:, 1:2, :, :] +
            0.114 * image[:, 2:3, :, :])


def _norm_Max(image):
    # normalize by using L_inf norm. This projects the color to the surface
    # of the RGB cube, where all multiples of 0-1 of it make sense and
    # reach all values in the cube (compared to normalizing with the mean)
    # where white cannot be recovered anymore since norm 1 gives grayish.
    return np.max(image, axis=1, keepdims=True)


def _threshold(image_intensity):
    return np.maximum(image_intensity, EPS)
