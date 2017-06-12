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

"""Layer that penalizes input values which are not between 0 and 1."""

from __future__ import absolute_import, division, print_function

import numpy as np
import sys
import os

try:
    import caffe
except ImportError:
    sys.path.insert(0, os.path.join(os.path.expanduser('~'),
                                    'Repositories',
                                    'malabar',
                                    'python'))
    import caffe


class BoundaryLossLayer(caffe.Layer):

    """Loss Layer penalizing values outside of the range 0-1."""

    def setup(self, bottom, top):
        """Check that layer is correctly set up by checking input."""
        if len(bottom) != 1:
            raise Exception("Expecting exactly one input.")
        if bottom[0].data.shape[1] != 3:
            raise Exception("Expecting an RGB (3 channel) image.")

        if self.param_str not in ['L1', 'L2']:
            self.param_str = 'L1'
            print("Using standard initialization in BoundaryLossLayer:",
                  self.param_str)

        # parse the params
        self.error_norm = getattr(self, self.param_str)

    def reshape(self, bottom, top):
        """Define dimensions of data."""
        # only scalar output
        top[0].reshape(1)

    def forward(self, bottom, top):
        """Forward pass of the layer."""
        # first compute the intensity
        intensity = np.mean(bottom[0].data, axis=1, keepdims=True)

        # now get pixelwise loss with either L1 or L2 norm
        loss, diff = self.error_norm(intensity)

        # final loss is mean of pixelwise loss
        top[0].data[...] = np.mean(loss)
        diff /= loss.size

        # go back to RGB diff
        self.diff = np.tile(diff/3, (1, 3, 1, 1))

    def backward(self, top, propagate_down, bottom):
        """Backward pass of the layer."""
        if propagate_down[0]:
            bottom[0].diff[...] = self.diff * top[0].diff[0]

    def L1(self, intensity):
        """Compute loss and derivative with L1 norm."""
        # prepare a pixelwise loss matrix
        loss = np.zeros_like(intensity)
        diff = np.zeros_like(intensity)

        # handle the entries < 0
        ids = intensity < 0
        loss[ids] = -intensity[ids]
        diff[ids] = -1

        # handle the entries > 1
        ids = intensity > 1
        loss[ids] = intensity[ids] - 1
        diff[ids] = 1

        return loss, diff

    def L2(self, intensity):
        """Compute loss and derivative with L2 norm."""
        # prepare a pixelwise loss matrix
        loss = np.zeros_like(intensity)
        diff = np.zeros_like(intensity)

        # handle the entries < 0
        ids = intensity < 0
        loss[ids] = intensity[ids]**2
        diff[ids] = 2 * intensity[ids]

        # handle the entries > 1
        ids = intensity > 1
        translation = intensity[ids] - 1
        loss[ids] = translation**2
        diff[ids] = 2 * translation

        return loss, diff
