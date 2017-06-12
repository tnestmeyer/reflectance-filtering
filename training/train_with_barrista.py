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
Full pipeline for training the WHDR CNN on IIW.

The helper script called by this one should just give the main idea how to
train the CNN proposed in the paper. The code is not at all clean. If you want
to retrain the CNN and you do not find enough guidance in it, please come back
to
thomas . nestmeyer at tuebingen . mpg . de
"""
from __future__ import print_function, division

import argparse
import sys
import os
import pprint
import platform
import shutil
import numpy as np

# the next two lines are to avoid showing the Rocket Symbol in OS X, see
# http://leancrew.com/all-this/2014/01/stopping-the-python-rocketship-icon/
import matplotlib
matplotlib.use("Agg")  # noqa

# set path of where to find barrista and import
# barrista_root = os.path.join(os.path.expanduser('~'),
#                              '.local/lib/python2.7/site-packages')
barrista_root = os.path.join(os.path.expanduser('~'),
                             'Repositories',
                             'classner_barrista')
sys.path.insert(0, barrista_root)
import barrista  # noqa  # importing now to use correct path although not used

# set path of malabar (caffe flavor) and import
malabar_root = os.path.join(os.path.expanduser('~'),
                            'Repositories',
                            'malabar')
py_root = os.path.join(malabar_root, 'python')
sys.path.insert(0, py_root)
import caffe  # noqa

# stop caffe from printing that much
DISPLAY_CAFFE = 0  # noqa
# DISPLAY_CAFFE = 1  # noqa


if DISPLAY_CAFFE:  # noqa
    os.environ["GLOG_minloglevel"] = "0"  # noqa
else:  # noqa
    os.environ["GLOG_minloglevel"] = "1"  # noqa


def mkdir_p(path):
    """
    Emulate 'mkdir -p'.

    Emulate 'mkdir -p' in python 2 where os.makedirs(path, exist_ok=True)
    is not available)
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    # in python 3 we could just do
    # os.makedirs(path, exist_ok=True)


def mkdirs(base_dir, create_dirs):
    """
    Create base_dir with subfolders in create_dirs.

    Create the folder base_dir (if not existing yet) and in it,
    create all subdirectories in create_dir.
    """
    mkdir_p(base_dir)
    for d in create_dirs:
        mkdir_p(os.path.join(base_dir, d))


def _vis_square(data, padsize=1, padval=0):
    """Inspired by the caffe ipython notebook on filter visualization."""
    data -= data.min()
    data /= data.max()
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) +
               ((0, 0),) * (data.ndim - 3))
    data = np.pad(data, padding, mode='constant',
                  constant_values=(padval, padval))
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) +
                        tuple(range(4, data.ndim + 1)))  # noqa
    data = data.reshape((n * data.shape[1], n * data.shape[3]) +
                        data.shape[4:])
    return data


# MAIN Method
if __name__ == "__main__":
    """Parse the input arguments from the command line."""
    # the above 'and' is to not try to start on sched!!!
    print("sys.argv:")
    pprint.pprint(sys.argv)
    parser = argparse.ArgumentParser(
        description="Parses the arguments and then runs the appropriate mode."
    )
    parser.add_argument("--stage", "-s",
                        dest="stage",
                        # required=True,
                        help="fit or predict")
    parser.add_argument("--iterations", "-i",
                        type=int,
                        # default=10000,
                        # required=True,
                        help="""number of iterations to train or with which
                                trained iteration to predict""")
    parser.add_argument("--solver",
                        dest="solverType",
                        default='ADAM',
                        help="ADAM or SGD")
    parser.add_argument("--base_lr", "-lr",
                        dest="base_lr",
                        type=float,
                        # SGD 0.01, ADAM 0.001 is a good starting point
                        default=0.001,
                        help="learning rate of the solver to use")
    parser.add_argument("--comparisonsType", "-comp",
                        dest="comparisonsType",
                        default='comparisons',
                        choices=['comparisons',
                                 'augmented'],
                        help="""to use the comparisons from IIW directly
                                (comparisons) or to use the transitive hull
                                of them (augmented).""")
    parser.add_argument("--networkType", "-net",
                        dest="networkType",
                        default='convStaticWithSigmoid',
                        choices=['uNet',
                                 'simpleConvolutionsRelu',
                                 'convStatic',
                                 'convIncreasing',
                                 'convStaticWithSigmoid',
                                 'convStaticSkipLayers',
                                 'cascadeSkipLayers',
                                 ],
                        help="networks to choose from")
    parser.add_argument("--loss_scale_whdr",
                        dest="loss_scale_whdr",
                        type=float,
                        default=10,
                        help="scaling of whdr loss")
    parser.add_argument("--loss_scale_lambert",
                        dest="loss_scale_lambert",
                        type=float,
                        default=0.0,
                        help="scaling of ||I-RS|| loss term")
    parser.add_argument("--shading_unary_type",
                        dest="shading_unary_type",
                        default='L1_0.5',
                        help="type of norm and barS for unary shading term")
    parser.add_argument("--loss_scale_boundaries01",
                        dest="loss_scale_boundaries01",
                        type=float,
                        default=0.1,
                        help="scaling of 0-1 boundaries enforcing loss term")
    parser.add_argument("--batch_size", "-b",
                        dest="batch_size",
                        type=int,
                        default=20,  # 40,
                        help="batch size to be used in training")
    parser.add_argument("--predictCaffemodel", "-pcm",
                        dest="predictCaffemodel",
                        default=None,
                        help="to directly predict for a certain caffemodel")
    parser.add_argument("--height",
                        dest="height",
                        type=int,
                        default=256,
                        help="what height for the images to use")
    parser.add_argument("--width",
                        dest="width",
                        type=int,
                        default=256,
                        help="what width for the images to use")
    parser.add_argument("--startOver",
                        dest="startOver",
                        type=int,
                        default=1,
                        help="If training should be run (again).")
    parser.add_argument("--alwaysComputeShadingLosses",
                        dest="alwaysComputeShadingLosses",
                        type=int,
                        default=0,
                        help="""Compute shading loss even when loss_weight=0
                                (this takes time, but can be shown in
                                visualization).""")
    parser.add_argument("--numLayers",
                        dest="numLayers",
                        type=int,
                        default=2,
                        help="How many 'inner' layers the net should have.")
    parser.add_argument("--RS_est_mode", "-RS",
                        dest="RS_est_mode",
                        default='rRelMax',
                        choices=['sAbs', 'S', 'rAbs', 'R', 'RS',
                                 'rRelNorm', 'rRelMean', 'rRelY', 'rRelMax',
                                 'sRelNorm', 'sRelMean', 'sRelY', 'sRelMax',
                                 'rDirectly',
                                 ],
                        help="""Semantic meaning of what the network should
                                predict. Scalar or RGB, shading or
                                reflectance.""")
    parser.add_argument("--kernel_pad",
                        dest="kernel_pad",
                        type=int,
                        default=1,
                        help="""Padding and resp. kernel size for convolutions.
                                1 for example uses a 3x3 kernel and pads 1
                                pixel at the border in order to keep the image
                                size, 0 is for the 1x1 kernel used in the
                                paper.""")
    parser.add_argument("--num_filters_log",
                        dest="num_filters_log",
                        type=int,
                        default=4,
                        help="""num_output to use in static convolution size.
                                if e.g. 5 is use, then 2^5=32 is num_output
                                """)
    parser.add_argument("--use_batch_normalization",
                        dest="use_batch_normalization",
                        type=int,
                        default=0,
                        help="""If batch normalization should be used.""")
    parser.add_argument("--checkpoint_interval",
                        dest="checkpoint_interval",
                        type=int,
                        default=1000,
                        help="After how many iterations to save a caffemodel.")
    parser.add_argument("--experiment", "-exp",
                        dest="experiment_name",
                        default='tmp',
                        help="Name of the experiment for subfolder.")
    parser.add_argument("--random_seed",
                        dest="random_seed",
                        type=int,
                        default=-1,
                        help="""barrista says: random_seed: int>0 or None.
                                But there is an error with None. proto says
                                the default is -1.
                                If specified, seeds the solver for reproducible
                                results. Otherwise, it uses a time dependent
                                seed.""")
    parser.add_argument("--dataset",
                        dest="dataset",
                        default='iiw',
                        choices=['iiw', 'sintel', 'mixed', 'nonsense'],
                        help="""Which dataset to use.""")
    parser.add_argument("--sRGB_linear",
                        dest="sRGB_linear",
                        default='linear',
                        choices=['sRGB', 'linear'],
                        help="""Provide input in sRGB or linear.""")
    parser.add_argument("--whdr_delta_margin_ratio_dense",
                        dest="whdr_delta_margin_ratio_dense",
                        default="0.1_0.05_1.0_1",
                        # default="0.11_0.08_1.0_1",
                        help="""- The delta to use in the WHDR Hinge Loss
                                - margin for the hinge
                                - what ratio of comparisons to evaluate
                                - if dense labels should be evaluated.""")
    parser.add_argument("--test",
                        dest="test",
                        type=int,
                        default=0,
                        help="""If the testset should be used (instead of
                                validation).""")
    parser.add_argument("--dilation",
                        dest="dilation",
                        type=int,
                        default=1,
                        help="""If the testset should be used (instead of
                                validation).""")
    parser.add_argument("--decompose",
                        dest="decompose",
                        action='append',
                        help="decompose images in a folder or a video")

    # args = parser.parse_args(sys.argv[1:])
    args = parser.parse_args()
    args.nodename = platform.node()
    print("Arguments:")
    pprint.pprint(vars(args))

    results_dir = os.path.join(os.path.expanduser('~'),
                               'Results',
                               args.experiment_name)
    mkdirs(results_dir, ['images',
                         'logs',
                         'networks',
                         'progressions',
                         'scores',
                         'framerates',
                         'snapshots',
                         'decompositions_linear',
                         'decompositions_sRGB',
                         ])

    if args.decompose:
        filename = os.path.join(results_dir,
                                'decompositions_linear',
                                '0command.txt')
        with open(filename, 'a') as command:
            # command.write(str(sys.argv) + "\n")
            for a in sys.argv:
                command.write(a + " ")
            command.write("\n")

        dst = os.path.join(results_dir,
                           'decompositions_sRGB',
                           '0command.txt')
        shutil.copy(filename, dst)

    from train_with_barrista_helper import fit_predict_net
    fit_predict_net(args, results_dir)
