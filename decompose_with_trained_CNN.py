#!/usr/bin/env python

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
Full intrinsic image decomposition pipeline for the WHDR CNN.

Image convention is always a linear RGB image with shape
channels x height x width in the range 0 - 1.
"""
from __future__ import print_function, division

import os
import sys
import argparse

import numpy as np

import image_utils as iu

caffe_path = os.path.join(os.path.expanduser('~'),
                          'Repositories',
                          'caffe',
                          'python')
# print("add caffe to python path:", caffe_path)
sys.path.insert(0, caffe_path)
try:
    import caffe
except ImportError:
    msg = ("No module named caffe. " +
           "Is 'caffe' on your python path? "
           "Point the variable 'caffe_path' above to the appropriate path." +
           "It is currentl set to: " + caffe_path)
    raise ImportError(msg)


def imgCV2_to_caffeBlob(img):
    """Take an OpenCV image and turn it into the caffe blob format."""
    # make range from 0-255 into 0-1
    blob = img / 255.0
    # change channels from BGR into RGB
    blob = blob[:, :, ::-1]
    # turn from sRGB into linear image
    blob = iu.srgb_to_rgb(blob)
    # change channels from h x w x c into c x h x w
    blob = np.transpose(blob, (2, 0, 1))
    # put blob into mini-batch of one image
    blob = blob[np.newaxis, :, :, :]
    return blob


def caffeBlob_to_imgGrayLinear(blob):
    """Take a caffe blob and turn it into an OpenCV image."""
    b, c = blob.shape[:2]
    if b != 1 or c != 1:
        msg = ("Expecting to get 1 image in mini-batch having 1 channel, " +
               "but got batch size of {} and {} channels".format(b, c))
        raise ValueError(msg)
    return blob[0, 0, :, :]


def get_reflectance_caffe(net, image):
    """Run the image through caffe and return result."""
    # prepare shape of input blob
    height, width = image.shape[:2]
    net.blobs['images'].reshape(1, 3, height, width)
    # set input blob
    net.blobs['images'].data[...] = imgCV2_to_caffeBlob(image)
    # do forward step
    net.forward()
    # get result from blobs
    reflectance_blob = net.blobs['reflectance_intensity'].data
    # extract image from it
    reflectance_gray = caffeBlob_to_imgGrayLinear(reflectance_blob)
    return reflectance_gray


def decompose_image(filename_in, path_out):
    """Run the intrinsic image decomposition with caffe."""
    network_file = os.path.join(os.path.dirname(__file__),
                                'network_definition.prototxt')
    caffemodel = os.path.join(os.path.dirname(__file__),
                              'learned_weights.caffemodel')
    net = caffe.Net(network_file,
                    caffe.TEST,
                    weights=caffemodel)

    # print("Read file:", filename_in)
    image = iu.imread(filename_in)

    # get basename for output later
    basename = os.path.splitext(os.path.basename(filename_in))[0]

    # get result from caffe
    reflectance_gray = get_reflectance_caffe(net, image)

    # save result
    filename = os.path.join(path_out, basename + '-r.png')
    iu.imwrite(filename, reflectance_gray)

    # now colorize with input image again
    reflectance, shading = iu.colorize(reflectance_gray, image)

    # save color versions in sRGB
    filename = os.path.join(path_out, basename + '-r_colorized.png')
    iu.imwrite(filename, reflectance, sRGB=True)
    filename = os.path.join(path_out, basename + '-s_colorized.png')
    iu.imwrite(filename, shading, sRGB=True)

    return reflectance_gray


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Decompose an image with the direct reflectance
                       prediction CNN."""
    )
    parser.add_argument("--filename_in",
                        help="""Filename of the image which should be
                                decomposed.""")
    parser.add_argument("--path_out",
                        help="""Where the resulting decompositions should be
                                saved.""")
    args = parser.parse_args()
    if args.filename_in and args.path_out:
        decompose_image(args.filename_in, args.path_out)
    else:
        parser.print_help()
