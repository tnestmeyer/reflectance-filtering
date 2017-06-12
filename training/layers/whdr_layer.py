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
WHDR layer.

Provides the layer that computes the WHDR Loss as explained in the intrinsic
images in the wild paper.
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import caffe  # needs to be on the path


class WhdrLayer(caffe.Layer):
    """
    WHDR Loss Layer.

    Compute the WHDR Loss as explained in the intrinsic images in the wild
    paper.
    """

    def setup(self, bottom, top):
        """Check that layer is correctly set up by checking input pair."""
        if len(bottom) != 2:
            raise Exception("Need two inputs, the reflectance image on which "
                            "the WHDR is supposed to be evaluated and the "
                            "ground truth comparisons.")

        params = self.param_str.split('_')
        if self.param_str == "":
            self.delta = 0.1  # threshold for "more or less equal"
        elif len(params) == 1:
            self.delta = float(params[0])
            assert(self.delta >= 0)
        else:
            msg = ("parameters to WhdrLayer were not as expected: " +
                   self.param_str +
                   " was provided, but need exactly one argument, " +
                   "the delta."
                   )
            raise Exception(msg)
        print("WhdrLayer uses delta =", self.delta)

    def reshape(self, bottom, top):
        """Define dimensions of data."""
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        """Forward pass of the layer."""
        batch_size, channels, height, width = bottom[0].data.shape
        whdr_sum = 0.0
        for b in range(batch_size):  # go through the batch
            # get the reflectance image
            refl_img = bottom[0].data[b, :, :, :]

            comparisons, file_name = self._get_comparisons(bottom, b,
                                                           height, width)
            whdr = self._whdr(refl_img, comparisons)
            whdr_sum += whdr

        # compute the final WHDR value as mean of the WHDRs in the batch
        whdr = whdr_sum / batch_size
        top[0].data[...] = whdr
        # print("WHDR sum for this batch is", whdr)

    def backward(self, top, propagate_down, bottom):
        """Backward pass of the layer."""
        if any(propagate_down):
            raise Exception("There is no proper gradient for the WHDR."
                            "Consider using the whdr hinge loss layer"
                            "that gives an approximation to the WHDR.")

    def _channel_weights(self, channels):
        return _channel_weights(channels)

    def _lightness(self, r):
        return _lightness(r)

    def _get_comparisons(self, bottom, b, height, width):
        return _get_comparisons_from_bottom(bottom, b,
                                            height, width,
                                            self.delta)

    def _whdr(self, reflectance, comparisons):
        return whdr(reflectance,
                    comparisons,
                    self.delta)

    def _visualize(self, whdr, refl_img, height, width, comparisons, fname):
        for index in range(self.num_imgs_in_vis):
            if whdr > self.whdrs[0][index]:
                if (index < self.num_imgs_in_vis-1 and
                        whdr > self.whdrs[0][index+1]):
                    continue
                px = index
                py = 0
                self.whdrs[py][0:px] = self.whdrs[py][1:px+1]
                self.whdrs[py][px] = whdr
                # print(py, px, i, whdr)

                img = cv2.normalize(refl_img[0, :, :],
                                    None, 0.0, 1.0, cv2.NORM_MINMAX)
                # show the comparisons by circles
                for c in range(comparisons.shape[0]):
                    x1, y1, x2, y2, darker, weight = comparisons[c, :]
                    cv2.circle(img, (x1, y1), 3, 1.0, 1)
                    cv2.circle(img, (x2, y2), 3, 1.0, 1)

                self.multiple_images[:height, 0:px*width] = \
                    self.multiple_images[:height, width:(px+1)*width]
                self.multiple_images[py*height:(py+1)*height,
                                     px*width:(px+1)*width] = img
                text = "{0:.3f} ".format(whdr) + str(int(fname))
                cv2.putText(self.multiple_images,           # image
                            text,                           # text
                            (px*width, (py+1)*height-20),   # origin
                            cv2.FONT_HERSHEY_SIMPLEX,       # font
                            0.5,                            # size
                            255)                            # color
                cv2.imshow('reflectance images sorted by WHDR',
                           self.multiple_images)
                cv2.waitKey(1)
                break
        for index in reversed(range(self.num_imgs_in_vis)):
            if whdr < self.whdrs[1][index]:
                if index > 0 and whdr < self.whdrs[1][index-1]:
                    continue
                px = index
                py = 1
                self.whdrs[py][px+1:] = self.whdrs[py][px:-1]
                self.whdrs[py][px] = whdr
                # print(py, px, i, whdr, self.whdrs[1])
                img = cv2.normalize(refl_img[0, :, :],
                                    None, 0.0, 1.0, cv2.NORM_MINMAX)
                # show the comparisons by circles
                for c in range(comparisons.shape[0]):
                    x1, y1, x2, y2, darker, weight = comparisons[c, :]
                    cv2.circle(img, (x1, y1), 3, 1.0, 1)
                    cv2.circle(img, (x2, y2), 3, 1.0, 1)
                self.multiple_images[height:, (px+1)*width:] = \
                    self.multiple_images[height:, px*width:-width]
                self.multiple_images[py*height:(py+1)*height,
                                     px*width:(px+1)*width] = img
                text = "{0:.3f} ".format(whdr) + str(int(fname))
                cv2.putText(self.multiple_images,           # image
                            text,                           # text
                            (px*width, (py+1)*height-20),   # origin
                            cv2.FONT_HERSHEY_SIMPLEX,       # font
                            0.5,                            # size
                            255)                            # color
                cv2.imshow('reflectance images sorted by WHDR',
                           self.multiple_images)
                cv2.waitKey(1)
                break


eps = np.finfo(np.float32).eps  # threshold


def _lightness(r):
    num_channels = len(r)
    if num_channels == 3:
        # use mean to compute lightness
        # without truncation:
        # L =  np.mean(r)
        # with truncation:
        L = max(eps, np.mean(r))
        # L = max(1e-10, np.mean(r))
        # derivative of the mean
        dLdR = np.ones(3) / 3.
    elif num_channels == 1:
        L = max(eps, r)
        dLdR = np.ones(1)
    else:
        raise Exception("Expecting 1 or 3 channels to compute lightness!")
    return L, dLdR


def _get_comparisons_from_bottom(bottom, b, height, width, delta):
    file_name = bottom[1].data[b, -1, 0, 1]
    # get the comparisons
    comparisons = bottom[1].data[b, :, 0, :]
    res = _extract_valid_comparisons_with_actual_size(comparisons,
                                                      height,
                                                      width)
    return res, file_name


def get_comparisons_from_blob(comp_blob, height, width, delta,
                              ground_truth_albedo=None):
    """
    Extract comparisons needed for the WHDR computation.

    Args:
        comp_blob: array of shape [MAX_NUM_COMPARISONS, 1, 6] containing
            comparisons (including invalid ones: NaN)
        height (int): The height to which y in 0-1 should be scaled
        width (int): The width to which x in 0-1 should be scaled
        delta (float): The delta parameter in [Bell 2014].

    Returns:
        [num_comparisons, 6] array: The first output consists of the
            comparisons for the WHDR and where num_comparisons is either the
            number of available comparisons in the corresponding image, or the
            number of comparisons which are dynamically created.
        float: The original file name in the IIW dataset.
    """
    file_name = comp_blob[-1, 0, 1]
    # get the comparisons
    comparisons = comp_blob[:, 0, :]
    res = _extract_valid_comparisons_with_actual_size(comparisons,
                                                      height,
                                                      width)
    return res, file_name


def _extract_valid_comparisons_with_actual_size(comparisons, height, width):
    num_comparisons = int(comparisons[-1, 0])
    # needs to be copied, because we will change 'res'
    res = comparisons[:num_comparisons, :].copy()
    # get the real coordinates by multiplying
    # x1 = int(x1*width)
    # y1 = int(y1*height)
    # x2 = int(x2*width)
    # y2 = int(y2*height)
    res[:, [0, 2]] = (res[:, [0, 2]] * width).astype(int)
    res[:, [1, 3]] = (res[:, [1, 3]] * height).astype(int)
    return res


def whdr(reflectance, comparisons, delta):
    """
    Compute WHDR as in [Bell 2014].

    Expects a reflectance image of shape [c, h, w] and
    a comparisons blob [num_only_valid_comparisons, 6].
    """
    error_sum = 0.0
    weight_sum = 0.0
    for c in range(comparisons.shape[0]):
        x1, y1, x2, y2, darker = comparisons[c, :5].astype(int)
        weight = comparisons[c, 5]

        # pay attention to the ordering: channel times y times x
        r1 = reflectance[:, y1, x1]
        r2 = reflectance[:, y2, x2]
        l1, _dummy = _lightness(r1)
        l2, _dummy = _lightness(r2)

        if l2/l1 > 1+delta:
            alg_darker = 1  # '1'
        elif l1/l2 > 1+delta:
            alg_darker = 2  # '2'
        else:
            alg_darker = 0  # 'E'
        if darker != alg_darker:
            error_sum += weight
        weight_sum += weight

    # catch a possible weight_sum = 0 if there are no comparisons
    if weight_sum:
        whdr = error_sum / weight_sum
    else:
        whdr = 0.0
    return whdr
