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
WHDR Hinge Loss layer.

Provides the layer that computes a Hinge loss approximation to the WHDR as
explained in the intrinsic images in the wild paper.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from whdr_layer import WhdrLayer


MAX_EVALUATED_COMPARISONS = 1500  # 2500  # non-augmented has max 1181



# derive from WhdrLayer to get its methods
# (get comparisons, visualize, lightness)
class WhdrHingeLossLayer(WhdrLayer):
    """
    WHDR Hinge Loss Layer.

    Compute a Hinge loss approximation to the WHDR Loss as explained in the
    intrinsic images in the wild paper.
    """

    def setup(self, bottom, top):
        """Check that layer is correctly set up by checking input pair."""
        if len(bottom) != 2 and len(bottom) != 3:
            raise Exception("Need two inputs, the reflectance image on which "
                            "the WHDR is supposed to be evaluated and the "
                            "ground truth comparisons. In the case of Sintel"
                            "a third input with the ground truth reflectances"
                            "is needed and the given comparisons should be 0.")
        params = self.param_str.split('_')
        if self.param_str == "":
            self.delta = 0.1  # threshold for "more or less equal"
            self.margin = 0.0  # margin in the Hinge
            self.ratio = 1.0  # ratio of evaluated comparisons
            self.eval_dense = 1  # evaluate dense labels?
        elif len(params) == 4:
            self.delta = float(params[0])
            assert(self.delta >= 0)
            self.margin = float(params[1])
            assert(self.margin >= 0)
            self.ratio = float(params[2])
            assert(0 < self.ratio <= 1)
            self.eval_dense = int(params[3])
        else:
            msg = ("parameters to WhdrHingeLossLayer were not as expected: " +
                   self.param_str +
                   " was provided, but need four arguments, " +
                   "delta, margin" +
                   "ratio of comparisons, if dense " +
                   "labels are supposed to be evaluated."
                   )
            raise Exception(msg)
        print("WhdrHingeLossLayer uses",
              "delta =", self.delta,
              "margin =", self.margin,
              "ratio of evaluated comparisons =", self.ratio,
              "evaluate dense labels:", self.eval_dense,
              )

    def reshape(self, bottom, top):
        """Define dimensions of data."""
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        """Forward pass of the layer."""
        # start = timeit.default_timer()
        batch_size, channels, height, width = bottom[0].data.shape

        # prepare blob for backprop step to share computation
        # (it is changed in self._whdr_hinge_single_img !!!)
        self.diff = np.zeros_like(bottom[0].data)

        whdrs = [self._whdr_hinge_single_img(bottom, b)
                 for b in range(batch_size)]

        # compute the final WHDR value as mean of the WHDRs in the batch
        whdr = np.mean(whdrs)
        # also apply the mean (division by batch_size) to gradient
        self.diff /= batch_size

        top[0].data[...] = whdr


    def backward(self, top, propagate_down, bottom):
        """Backward pass of the layer."""
        # compute gradient to (reflectance) image
        if propagate_down[0]:
            loss_weight = top[0].diff[0]
            # derivative is mostly computed in forward
            bottom[0].diff[...] = self.diff * loss_weight

        # check that there is no gradient to the ground truth computed
        if propagate_down[1]:
            print('Layer cannot backpropagate to ground truth comparisons.')
            # raise Exception('Layer cannot backpropagate to ground truth.')

    def _whdr_hinge_single_img(self, bottom, b):
        # inner_start = timeit.default_timer()

        # get the reflectance image
        refl_img = bottom[0].data[b, :, :, :]
        height, width = refl_img.shape[1:]
        comparisons, file_name = self._get_comparisons(bottom, b,
                                                       height, width)

        num_comparisons = comparisons.shape[0]
        if not self.eval_dense and num_comparisons > 300:
            # do not evaluate densely annotated images
            num_comparisons = 1
        if self.ratio < 1.0:
            num_comparisons = int(np.ceil(self.ratio * num_comparisons))

        if num_comparisons <= MAX_EVALUATED_COMPARISONS:
            # for non-augmented data this should always be the case
            comp_list = range(num_comparisons)
        else:
            comp_list = np.random.choice(num_comparisons,
                                         MAX_EVALUATED_COMPARISONS,
                                         replace=False)
        weight_sum = np.sum(comparisons[comp_list, 5])
        error_sum = sum([self._eval_single_comparison(refl_img,
                                                      comparisons[c, :],
                                                      b)
                         for c in comp_list])

        # catch a possible weight_sum = 0 if there are no comparisons
        if weight_sum:
            whdr = error_sum / weight_sum
            self.diff[b, :, :, :] /= weight_sum
        else:
            whdr = 0.0

        return whdr

    def _eval_single_comparison(self, refl_img, comparison, b):
        x1, y1, x2, y2, darker = comparison[0:5].astype(int)
        weight = comparison[5]

        # pay attention to the blob ordering:
        # channel times y times x
        R1 = refl_img[:, y1, x1]
        R2 = refl_img[:, y2, x2]
        # get the lightness instead of the (r,g,b) and gradient
        L1, dL1dR = self._lightness(R1)
        L2, dL2dR = self._lightness(R2)
        # compute ratio once
        L2inv = 1. / L2
        y = L1 * L2inv      # => y = L1 / L2
        # derivative of the ratio
        dydL1 = L2inv       # => dydL1 = 1. / L2
        dydL2 = -y * L2inv  # => dydL2 = -L1 / L2**2

        # branch into the cases 0(=E), 1, 2
        if darker == 1:  # L1 is darker than L2
            border = 1 / (1 + self.delta + self.margin)
            if y > border:
                loss_y = y - border   # loss_y = max(0, y - border)
                dldy = 1  # derivative of the hinge loss
            else:
                loss_y = 0
                dldy = 0
        elif darker == 2:  # L2 is darker than L1
            border = 1 + self.delta + self.margin
            if y < border:
                loss_y = border - y  # loss_y = max(0, border - y)
                dldy = -1
            else:
                loss_y = 0
                dldy = 0
        elif darker == 0:  # L1 and L2 are more or less the same
            if self.margin <= self.delta:
                # this should normally be the case that makes sense
                border_right = 1 + self.delta - self.margin
                # loss_y = max(0, border_left - y, y - border_right)
                if y > border_right:
                    loss_y = y - border_right
                    dldy = 1
                else:
                    border_left = 1 / border_right
                    if y < border_left:
                        loss_y = border_left - y
                        dldy = -1
                    else:
                        loss_y = 0
                        dldy = 0
            else:
                border = 1 + self.delta - self.margin
                loss_y = max(1/border - y, y - border)
                if y > 1:
                    dldy = 1
                else:
                    dldy = -1
        else:
            raise Exception('darker is neither 0(=E), 1, 2')
        error = weight * loss_y

        # the final derivatives by chain rule
        self.diff[b, :, y1, x1] += weight * dldy * dydL1 * dL1dR
        self.diff[b, :, y2, x2] += weight * dldy * dydL2 * dL2dR

        return error
