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

"""Go through color and spatial parameters and evaluate filter."""

from __future__ import print_function, division

import sys
import os
import argparse

# if your OpenCV installation is not in the python path, then
# use the next line(s) to add its path, otherwise just comment the insert.
# IMPORTANT: we need the ximgproc (Extended Image Processing) module of
# OpenCV here!
sys.path.insert(0, os.path.join(os.path.expanduser('~'),
                                '.local',
                                'lib',
                                'opencv3.1.0',
                                'lib',
                                'python2.7',
                                'dist-packages'))
import cv2  # noqa

import image_utils as iu


def apply_filter(filter_type, image, joint, sigma_color, sigma_spatial):
    """
    Apply the joint/guided filter.

    Apply the filter of the given type on the image according to the
    joint/guidance image and based on the given parameters.
    """
    if sigma_color <= 0 or sigma_spatial <= 0:
        raise ValueError("Parameters are expected to be positive.")
    if filter_type == 'bilateral':
        # using the joint bilateral filter instead
        filtered = cv2.ximgproc.jointBilateralFilter(joint,
                                                     image,
                                                     d=-1,
                                                     sigmaColor=sigma_color,
                                                     sigmaSpace=sigma_spatial)
    elif filter_type == 'guided':
        # or the guided filter
        filtered = cv2.ximgproc.guidedFilter(guide=joint,
                                             src=image,
                                             radius=int(sigma_spatial),
                                             eps=sigma_color)
    else:
        raise ValueError("filter_type must be 'bilateral' or 'guided'.")
    return filtered


def read_filter_write(filter_type,
                      filename_in, guidance_in,
                      sigma_color, sigma_spatial,
                      path_out):
    """Read input and guidance image, apply filter and write result."""
    # get basename for output later
    basename = os.path.splitext(os.path.basename(filename_in))[0]
    # Read the images
    image = iu.imread(filename_in)
    joint = iu.imread(guidance_in)

    filtered = apply_filter(filter_type,
                            image, joint,
                            sigma_color, sigma_spatial)

    # save the result
    params = "_{}_c{}s{}".format(filter_type, sigma_color, sigma_spatial)
    filename = os.path.join(path_out, basename + params + '.png')
    iu.imwrite(filename, filtered)

    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Filter reflectance prediction with a bilateral/guided
                       filter, to enhance piecewise constant reflectance
                       prior."""
    )
    parser.add_argument("--filename_in",
                        help="""Filename of the image which should be
                                filtered.""")
    parser.add_argument("--guidance_in",
                        help="""Filename of the guidance image which should be
                                used for filtering.""")
    parser.add_argument("--path_out",
                        help="""Where the resulting decompositions should be
                                saved.""")
    parser.add_argument("--sigma_color",
                        type=float,
                        help="color parameter")
    parser.add_argument("--sigma_spatial",
                        type=float,
                        help="spatial parameter")
    parser.add_argument("--filter_type",
                        help="""Which filter to choose,
                                the guided filter (guided) or
                                the joint bilateral filter (bilateral).""")

    args = parser.parse_args()
    if len(sys.argv) > 1:
        read_filter_write(args.filter_type,
                          args.filename_in, args.guidance_in,
                          args.sigma_color, args.sigma_spatial,
                          args.path_out)
    else:
        parser.print_help()
        print("If you do not have any idea what parameters to choose, " +
              "try one of the following combinations:")
        # for filtering the direct CNN prediction with itself
        print("--filter_type=bilateral --sigma_color=20 --sigma_spatial=22")
        print("--filter_type=guided --sigma_color=7 --sigma_spatial=52")
        # for filtering the direct CNN prediction with 'flat'
        print("--filter_type=guided --sigma_color=3 --sigma_spatial=45")
