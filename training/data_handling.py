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

"""Provide some general tools that are needed in different files."""
from __future__ import print_function, division

import os
import sys
import timeit
from time import sleep
import numpy as np


# provide the getData functionality
def getData(dataset,
            description,
            comparisonsType='comparisons'):
    """load Data from numpy file and return images and comparisons."""
    # print(dataset, description, comparisonsType)
    start = timeit.default_timer()
    folder = os.path.join(os.path.expanduser('~'),
                          'LMDBs',
                          dataset)
    filename = description + '.npz'
    full_path = os.path.join(folder, filename)
    if not os.path.isfile(full_path):
        raise IOError("File {} could not be found.".format(full_path))

    if os.stat(full_path).st_size > 1024*1024*100:
        print("Loading file", full_path, 'takes some time.')
        # flush the stdout (write content to file in cluster)
        # to immediately see output
        sys.stdout.flush()

    successfully_read_data = False
    counter = 0
    data = {}
    while counter < 10 and not successfully_read_data:
        try:
            with np.load(full_path) as npzFile:
                # for kind in ['images', 'comparisons', 'augmented']:
                #     data[kind] = npzFile[kind]

                for kind in ['images', 'comparisons']:
                    data[kind] = npzFile[kind]
                # data['description'] = "{}_{}".format(dataset, description)
                if comparisonsType == 'augmented':
                    data['augmented'] = npzFile['augmented']
                if dataset == 'sintel' or dataset == 'mixed':
                    data['albedos'] = npzFile['albedos']
            successfully_read_data = True
        except MemoryError:
            sec = np.random.rand() * 60  # try again up to a min later
            print("Reading of data was not successfull, trying again in",
                  sec, "seconds")
            sleep(sec)
            data = {}
            counter += 1
    stop = timeit.default_timer()
    print("Time needed to load data", description,
          "from dataset", dataset,
          "is:", stop-start, "seconds.")
    # flush the stdout (write content to file in cluster) for debugging
    sys.stdout.flush()
    return data


def rgb_to_srgb(rgb):
    """Taken from bell2014: RGB -> sRGB."""
    ret = np.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = np.power(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret


def srgb_to_rgb(srgb):
    """Taken from bell2014: sRGB -> RGB."""
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret
