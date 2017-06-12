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

"""Different network definitions."""
from __future__ import print_function, division

import os
import sys
import cv2

import barrista
import barrista.design as design
from barrista.design import (ConcatLayer,
                             ConvolutionLayer,
                             DeconvolutionLayer,
                             DropoutLayer,
                             EltwiseLayer,
                             EuclideanLossLayer,
                             InterpLayer,
                             PoolingLayer,
                             PowerLayer,
                             PReLULayer,
                             PythonLayer,
                             ReLULayer,
                             ScaleLayer,
                             SliceLayer,
                             SigmoidLayer,
                             # SilenceLayer,
                             PROTODETAIL)

# tell python where to find the python layers
layers_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                          'layers')
sys.path.insert(0, layers_path)


NUM_COMPARISONS = 1181
NUM_AUGMENTED = 60049  # 89546


# print("Imported barrista from", barrista.__path__)

def create_network(args, draw_net_filename=None):
    """Call creation of a specific network defined by networkType."""
    images_shape = [args.batch_size, 3, args.height, args.width]
    comparisons_shape = [args.batch_size, NUM_COMPARISONS+1, 1, 6]

    input_shapes = [images_shape, comparisons_shape]
    inputs = ['images', 'comparisons']
    if args.comparisonsType == 'augmented':
        augmented_shape = [args.batch_size, NUM_AUGMENTED+1, 1, 6]
        input_shapes.append(augmented_shape)
        inputs.append('augmented')
    if args.dataset == 'sintel':
        input_shapes.append(images_shape)
        inputs.append('albedos')

    predict_inputs = ['images']
    predict_input_shapes = [[1, 3, args.height, args.width]]
    print("Network input:", input_shapes, inputs)
    netspec = design.NetSpecification(
        input_shapes,
        inputs=inputs,
        predict_inputs=predict_inputs,
        predict_input_shapes=predict_input_shapes,
        name='Networks_in_barrista'
    )

    # define the filler type:
    args.filler_type = "xavier"
    # args.filler_type = "constant"

    # possible filler types: "constant", "gaussian", "positive_unitball",
    #                        "uniform", "xavier", "msra", "bilinear"

    RS_est_mode = args.RS_est_mode.split('-')[0]
    if RS_est_mode in ['RS']:
        # if we want to estimate R and S at the same time
        num_output = 6
    elif RS_est_mode in ['S', 'R']:
        # if we want to use a RGB estimation
        num_output = 3
    elif RS_est_mode in ['sAbs', 'rAbs',
                         'rRelNorm', 'rRelMean', 'rRelY', 'rRelMax',
                         'sRelNorm', 'sRelMean', 'sRelY', 'sRelMax',
                         'rDirectly',
                         ]:
        # if we want to use a scalar estimation
        num_output = 1
    else:
        msg = "RS-estimation '{}' not known".format(RS_est_mode)
        raise Exception(msg)

    # get the network dependent on the name
    function = 'create_' + args.networkType
    layers = globals()[function](args, num_output)
    # interface: every network architecture has to end with tops=['RS_est']

    # recover reflectance and shading from the estimation
    layers.extend(recover_reflectance_shading(args, num_output))

    # add the loss layers to all the networks
    layers.extend(add_loss_layers(args))

    # # show output for debugging
    # layers.append(PythonLayer(Python_module='save_img_layer',
    #                           Python_layer='SaveImgLayer',
    #                           # Python_param_str="0.1",
    #                           name='save_images',
    #                           bottoms=['images',
    #                                    'reflectance',
    #                                    ],
    #                         #   bottoms=['images',
    #                         #            'refl_bi',
    #                         #            'reflectance_level0',
    #                         #            'reflectance',
    #                         #            ],
    #                           tops=[],
    #                           ))

    netspec.layers.extend(layers)
    # Create the network. Notice how all layers are automatically wired! If you
    # selectively name layers or blobs, this is taken into account.

    # print("Now instantiate the net")
    net = netspec.instantiate()
    # print("Net was instantiated")

    if draw_net_filename is not None:
        viz = netspec.visualize()
        cv2.imwrite(draw_net_filename, viz)
        prototxt = draw_net_filename[:-3] + 'prototxt'
        netspec.to_prototxt(output_filename=prototxt)

    return net

# from the layer catalogue:
# convolution parameters are:
# Required
# num_output, kernel_size (or kernel_h and kernel_w)
# Strongly Recommended
# weight_filler [default type: 'constant' value: 0]
# Optional
# bias_term [default true],
# pad (or pad_h and pad_w) [default 0],
# stride (or stride_h and stride_w) [default 1]
# group (g) [default 1]


def recover_reflectance_shading(args, num_output_final):
    """Recover reflectance and shading from estimation."""
    layers = []

    RS_est_mode = args.RS_est_mode.split('-')[0]
    if RS_est_mode in ['RS']:
        # if we want to estimate R and S at the same time, nothing needs
        # to be recovered, just directly give R and S
        layers.append(
            SliceLayer(
                name='slice_RS',
                Slice_axis=1,
                Slice_slice_point=[3],
                bottoms=['RS_est'],
                tops=['reflectance', 'shading'],
            )
        )
        return layers  # already return here
    elif RS_est_mode in ['rDirectly']:
        layers.append(
            ReLULayer(
                name='pass_on_r_to_reflectance',
                bottoms=['RS_est'],
                tops=['reflectance'],
            )
        )
        layers.append(
            ReLULayer(
                name='pass_on_r_to_shading_dummy',
                bottoms=['RS_est'],
                tops=['shading'],
            )
        )
        return layers  # already return here

    estimation = 'RS_est'

    # recover reflectance and shading:
    layers.append(
        PythonLayer(
            Python_module='recover_reflectance_shading_layer',
            Python_layer='RecoverReflectanceShadingLayer',
            Python_param_str=args.RS_est_mode,
            name='recover_reflectance_shading',
            bottoms=[estimation,
                     'images'],
            tops=['reflectance',
                  'shading']
        )
    )
    return layers


def add_loss_layers(args):
    """Define the loss layers."""
    layers = []
    # compute whdr hinge loss
    bottoms = ['reflectance', args.comparisonsType]
    if args.dataset == 'sintel':
        bottoms.append('albedos')
    layers.append(
        PythonLayer(
            Python_module='whdr_hinge_loss_layer',
            Python_layer='WhdrHingeLossLayer',
            Python_param_str=(args.whdr_delta_margin_ratio_dense),
            name='loss_whdr_hinge',
            bottoms=bottoms,
            tops=['loss_whdr_hinge'],
            loss_weights=[args.loss_scale_whdr],
            include_stages=['fit']
        )
    )

    # compute 'real' WHDR as 'accuracy' for evaluation
    bottoms = ['reflectance', 'comparisons']
    if args.dataset == 'sintel':
        bottoms.append('albedos')
    layers.append(PythonLayer(Python_module='whdr_layer',
                              Python_layer='WhdrLayer',
                              Python_param_str="0.1",
                              name='whdr_original',
                              bottoms=bottoms,
                              tops=['whdr_original'],
                              # do not account as loss layer:
                              loss_weights=[0],
                              include_stages=['fit']))

    if args.loss_scale_boundaries01 and args.RS_est_mode != 'rDirectly':
        layers.append(
            PythonLayer(Python_module='boundary_loss_layer',
                        Python_layer='BoundaryLossLayer',
                        Python_param_str=args.shading_unary_type[:2],
                        name='loss_boundaries_reflectance',
                        bottoms=['reflectance'],
                        tops=['loss_boundaries_reflectance'],
                        # loss layer
                        loss_weights=[args.loss_scale_boundaries01],
                        include_stages=['fit'])
        )
        layers.append(
            PythonLayer(Python_module='boundary_loss_layer',
                        Python_layer='BoundaryLossLayer',
                        Python_param_str=args.shading_unary_type[:2],
                        name='loss_boundaries_shading',
                        bottoms=['shading'],
                        tops=['loss_boundaries_shading'],
                        # loss layer
                        loss_weights=[args.loss_scale_boundaries01],
                        include_stages=['fit'])
        )

    RS_est_mode = args.RS_est_mode.split('-')[0]
    if RS_est_mode == 'RS':  # add the lambertian term
        layers.append(
            EltwiseLayer(
                bottoms=['reflectance', 'shading'],
                tops=['lambert'],
                # PROD = 0; SUM = 1; MAX = 2;
                Eltwise_operation=0,
                include_stages=['fit']
            )
        )

        layers.append(
            EuclideanLossLayer(
                bottoms=['lambert', 'images'],
                tops=['loss_lambert'],
                loss_weights=[args.loss_scale_lambert],
                include_stages=['fit']
            )
        )

    return layers


def create_uNet(args, num_output_final):
    """Create network in barrista.

    Create a network in barrista similar to a combination of U-Net and
    the one in the learning data-driven reflectance priors paper.
    """
    print("Using u-net")
    layers = []

    filler = design.PROTODETAIL.FillerParameter()
    filler.type = args.filler_type

    kernel = 2 * args.kernel_pad + 1
    pad = args.kernel_pad

    # going down with local features in the U
    layers.append(ConvolutionLayer(Convolution_num_output=16,
                                   Convolution_kernel_size=3,
                                   Convolution_stride=2,
                                   Convolution_pad=1,
                                   Convolution_weight_filler=filler,
                                   name='Conv1',
                                   bottoms=['images']))
    for i in range(args.numLayers):
        layers.append(ReLULayer())
        layers.append(ConvolutionLayer(Convolution_num_output=16,
                                       Convolution_kernel_size=kernel,
                                       Convolution_pad=pad,
                                       Convolution_weight_filler=filler))

    layers.append(ReLULayer(tops=['L1']))
    layers.append(ConvolutionLayer(Convolution_num_output=32,
                                   Convolution_kernel_size=3,
                                   Convolution_stride=2,
                                   Convolution_pad=1,
                                   Convolution_weight_filler=filler,
                                   name='Conv2'))
    for i in range(args.numLayers):
        layers.append(ReLULayer())
        layers.append(ConvolutionLayer(Convolution_num_output=32,
                                       Convolution_kernel_size=kernel,
                                       Convolution_pad=pad,
                                       Convolution_weight_filler=filler))
    layers.append(ReLULayer(tops=['L2']))
    layers.append(ConvolutionLayer(Convolution_num_output=64,
                                   Convolution_kernel_size=3,
                                   Convolution_stride=2,
                                   Convolution_pad=1,
                                   Convolution_weight_filler=filler,
                                   name='Conv3'))
    for i in range(args.numLayers):
        layers.append(ReLULayer())
        layers.append(ConvolutionLayer(Convolution_num_output=64,
                                       Convolution_kernel_size=kernel,
                                       Convolution_pad=pad,
                                       Convolution_weight_filler=filler))
    layers.append(ReLULayer(tops=['L3']))
    layers.append(ConvolutionLayer(Convolution_num_output=64,
                                   Convolution_kernel_size=7,
                                   Convolution_stride=1,
                                   Convolution_pad=3,
                                   Convolution_weight_filler=filler,
                                   name='Conv4'))
    for i in range(args.numLayers):
        layers.append(ReLULayer())
        layers.append(ConvolutionLayer(Convolution_num_output=64,
                                       Convolution_kernel_size=kernel,
                                       Convolution_pad=pad,
                                       Convolution_weight_filler=filler))
    layers.append(ReLULayer(tops=['local']))

    # lower path with the image global features
    layers.append(PythonLayer(Python_module='resize_layer',
                              Python_layer='ResizeLayer',
                              name='resize',
                              bottoms=['images'],
                              tops=['resized']))

    layers.append(ConvolutionLayer(Convolution_num_output=32,
                                   Convolution_kernel_size=5,
                                   Convolution_stride=4,
                                   Convolution_pad=2,
                                   Convolution_weight_filler=filler,
                                   name='Conv5',
                                   bottoms=['resized']))
    # for i in range(args.numLayers):
    #     layers.append(ReLULayer())
    #     layers.append(ConvolutionLayer(Convolution_num_output=32,
    #                                    Convolution_kernel_size=5,
    #                                    Convolution_pad=2,
    #                                    Convolution_weight_filler=filler))
    layers.append(ReLULayer())
    layers.append(ConvolutionLayer(Convolution_num_output=32,
                                   Convolution_kernel_size=5,
                                   Convolution_stride=4,
                                   Convolution_pad=2,
                                   Convolution_weight_filler=filler,
                                   name='Conv6'))
    # for i in range(args.numLayers):
    #     layers.append(ReLULayer())
    #     layers.append(ConvolutionLayer(Convolution_num_output=32,
    #                                    Convolution_kernel_size=5,
    #                                    Convolution_pad=2,
    #                                    Convolution_weight_filler=filler))
    layers.append(ReLULayer())
    layers.append(ConvolutionLayer(Convolution_num_output=32,
                                   Convolution_kernel_size=5,
                                   Convolution_stride=4,
                                   Convolution_pad=2,
                                   Convolution_weight_filler=filler,
                                   name='Conv7'))
    # for i in range(args.numLayers):
    #     layers.append(ReLULayer())
    #     layers.append(ConvolutionLayer(Convolution_num_output=32,
    #                                    Convolution_kernel_size=5,
    #                                    Convolution_pad=2,
    #                                    Convolution_weight_filler=filler))
    layers.append(ReLULayer())
    layers.append(ConvolutionLayer(Convolution_num_output=64,
                                   Convolution_kernel_size=3,
                                   Convolution_stride=1,
                                   Convolution_pad=0,
                                   Convolution_weight_filler=filler,
                                   name='Conv8'))
    layers.append(ReLULayer(tops=['global_1']))

    # combine the local and global features
    layers.append(PythonLayer(Python_module='broadcast_layer',
                              Python_layer='BroadcastLayer',
                              name='broadcast',
                              bottoms=['global_1', 'local'],
                              tops=['global']))
    layers.append(ConcatLayer(bottoms=['local', 'global'],
                              name='Concatenate'))
    for i in range(args.numLayers):
        layers.append(ConvolutionLayer(Convolution_num_output=64,
                                       Convolution_kernel_size=kernel,
                                       Convolution_pad=pad,
                                       Convolution_weight_filler=filler))
        layers.append(ReLULayer())
    layers.append(ConvolutionLayer(Convolution_num_output=64,
                                   Convolution_kernel_size=3,
                                   Convolution_pad=1,
                                   Convolution_weight_filler=filler))
    layers.append(ReLULayer(tops=['R3']))

    # going up again
    layers.append(DeconvolutionLayer(Convolution_num_output=64,
                                     Convolution_kernel_size=2,
                                     Convolution_stride=2,
                                     Convolution_weight_filler=filler,
                                     tops=['R3d']))
    layers.append(ConcatLayer(bottoms=['L2', 'R3d']))
    for i in range(args.numLayers):
        layers.append(ConvolutionLayer(Convolution_num_output=32,
                                       Convolution_kernel_size=kernel,
                                       Convolution_pad=pad,
                                       Convolution_weight_filler=filler))
        layers.append(ReLULayer())
    layers.append(ConvolutionLayer(Convolution_num_output=32,
                                   Convolution_kernel_size=3,
                                   Convolution_pad=1,
                                   Convolution_weight_filler=filler))
    layers.append(ReLULayer(tops=['R2']))

    layers.append(DeconvolutionLayer(Convolution_num_output=16,
                                     Convolution_kernel_size=2,
                                     Convolution_stride=2,
                                     Convolution_weight_filler=filler,
                                     tops=['R2d']))
    layers.append(ConcatLayer(bottoms=['L1', 'R2d']))
    for i in range(args.numLayers):
        layers.append(ConvolutionLayer(Convolution_num_output=16,
                                       Convolution_kernel_size=kernel,
                                       Convolution_pad=pad,
                                       Convolution_weight_filler=filler))
        layers.append(ReLULayer())
    layers.append(ConvolutionLayer(Convolution_num_output=16,
                                   Convolution_kernel_size=3,
                                   Convolution_pad=1,
                                   Convolution_weight_filler=filler))
    layers.append(ReLULayer(tops=['R1']))

    layers.append(DeconvolutionLayer(Convolution_num_output=3,
                                     Convolution_kernel_size=2,
                                     Convolution_stride=2,
                                     Convolution_weight_filler=filler,
                                     tops=['R1d']))
    layers.append(ConcatLayer(bottoms=['images', 'R1d']))

    for i in range(args.numLayers):
        layers.append(ConvolutionLayer(Convolution_num_output=3,
                                       Convolution_kernel_size=kernel,
                                       Convolution_pad=pad,
                                       Convolution_weight_filler=filler))
        layers.append(ReLULayer())

    layers.append(ConvolutionLayer(Convolution_num_output=num_output_final,
                                   Convolution_kernel_size=3,
                                   Convolution_pad=1,
                                   Convolution_weight_filler=filler,
                                   tops=['RS_est']))

    return layers


def create_simpleConvolutionsRelu(args, num_output_final):
    """Create network in barrista."""
    print("Using simple convolutions network with ReLU units, num_output is",
          "16, 32 *", args.numLayers, ", 16, 1 and a final Sigmoid.",
          "kernel_pad is", args.kernel_pad)
    layers = []

    filler = design.PROTODETAIL.FillerParameter()
    filler.type = args.filler_type
    # filler.value = 1

    kernel = 2 * args.kernel_pad + 1
    pad = args.kernel_pad

    layers.append(ConvolutionLayer(Convolution_kernel_size=kernel,
                                   Convolution_num_output=16,
                                   Convolution_stride=1,
                                   Convolution_pad=pad,
                                   Convolution_weight_filler=filler,
                                   bottoms=['images']))
    layers.append(ReLULayer())

    for i in range(args.numLayers):
        layers.append(ConvolutionLayer(Convolution_kernel_size=kernel,
                                       Convolution_num_output=32,
                                       Convolution_stride=1,
                                       Convolution_pad=pad,
                                       Convolution_weight_filler=filler))
        layers.append(ReLULayer())

    layers.append(ConvolutionLayer(Convolution_kernel_size=kernel,
                                   Convolution_num_output=16,
                                   Convolution_stride=1,
                                   Convolution_pad=pad,
                                   Convolution_weight_filler=filler))
    layers.append(ReLULayer())

    layers.append(ConvolutionLayer(Convolution_kernel_size=kernel,
                                   Convolution_num_output=num_output_final,
                                   Convolution_pad=pad,
                                   Convolution_weight_filler=filler,
                                   tops=['RS_est']))

    return layers


def create_convStatic(args, num_output_final):
    """Create network in barrista."""
    layers = []

    do_batch_normalization = False
    # if you want to use a Prelu, do the following below:
    # layers.append(PReLULayer())
    # or SigmoidLayer

    filler = design.PROTODETAIL.FillerParameter()
    filler.type = args.filler_type
    # filler.value = 1

    kernel = 2 * args.kernel_pad + 1
    pad = args.kernel_pad + (args.dilation - 1)
    num_output = 2**args.num_filters_log
    print("Using a convolutional network with ReLU units.",
          "It has", args.numLayers, "layers with",
          num_output, "filters each, then a conv layer with num_output",
          "filters (dependent on the RS_estimation mode)",
          "The kernels are of size", kernel, "with a padding of", pad)

    if args.numLayers >= 1:
        layers.append(
            ConvolutionLayer(
                name='conv0',
                Convolution_num_output=num_output,
                Convolution_kernel_size=kernel,
                Convolution_pad=pad,
                Convolution_dilation=args.dilation,
                Convolution_weight_filler=filler,
                bottoms=['images']
            )
        )
        if do_batch_normalization:
            layers.append(BatchNormLayer())
        layers.append(ReLULayer())

        for i in range(1, args.numLayers):
            layers.append(
                ConvolutionLayer(
                    name='conv{}'.format(i),
                    Convolution_num_output=num_output,
                    Convolution_kernel_size=kernel,
                    Convolution_pad=pad,
                    Convolution_dilation=args.dilation,
                    Convolution_weight_filler=filler
                )
            )
            if do_batch_normalization:
                layers.append(BatchNormLayer())
            layers.append(ReLULayer())

        layers.append(
            ConvolutionLayer(
                name='conv{}'.format(i+1),
                Convolution_num_output=num_output_final,
                Convolution_kernel_size=1,
                Convolution_pad=0,
                Convolution_weight_filler=filler,
                tops=['RS_est']
            )
        )
    else:
        # catch dummy case of numLayers == 0
        layers.append(
            ConvolutionLayer(
                name='conv0',
                Convolution_num_output=num_output_final,
                Convolution_kernel_size=kernel,
                Convolution_pad=pad,
                Convolution_dilation=args.dilation,
                Convolution_weight_filler=filler,
                bottoms=['images'],
                tops=['RS_est']
            )
        )

    return layers


def create_convStaticWithSigmoid(args, num_output_final):
    """Create network in barrista."""
    layers = []

    do_batch_normalization = False
    # if you want to use a Prelu, do the following below:
    # layers.append(PReLULayer())
    # or SigmoidLayer

    filler = design.PROTODETAIL.FillerParameter()
    filler.type = args.filler_type
    # filler.value = 1

    kernel = 2 * args.kernel_pad + 1
    pad = args.kernel_pad + (args.dilation - 1)
    num_output = 2**args.num_filters_log
    print("Using CNN with ReLU units and sigmoid in the end.",
          "It has", args.numLayers, "layers with",
          num_output, "filters each, then a conv layer with num_output",
          "filters (dependent on the RS_estimation mode).",
          "The kernels are of size", kernel, "with a padding of", pad)

    if args.numLayers >= 1:
        layers.append(
            ConvolutionLayer(
                name='conv0',
                Convolution_num_output=num_output,
                Convolution_kernel_size=kernel,
                Convolution_pad=pad,
                Convolution_dilation=args.dilation,
                Convolution_weight_filler=filler,
                bottoms=['images']
            )
        )
        if do_batch_normalization:
            layers.append(BatchNormLayer())
        layers.append(ReLULayer())

        for i in range(1, args.numLayers):
            layers.append(
                ConvolutionLayer(
                    name='conv{}'.format(i),
                    Convolution_num_output=num_output,
                    Convolution_kernel_size=kernel,
                    Convolution_pad=pad,
                    Convolution_dilation=args.dilation,
                    Convolution_weight_filler=filler
                )
            )
            if do_batch_normalization:
                layers.append(BatchNormLayer())
            layers.append(ReLULayer())

        layers.append(
            ConvolutionLayer(
                name='conv{}'.format(i+1),
                Convolution_num_output=num_output_final,
                Convolution_kernel_size=1,
                Convolution_pad=0,
                Convolution_weight_filler=filler,
                tops=['RS_est_before_sigmoid']
            )
        )
        layers.append(SigmoidLayer(bottoms=['RS_est_before_sigmoid'],
                                   tops=['RS_est']))
    else:
        # catch dummy case of numLayers == 0
        layers.append(
            ConvolutionLayer(
                name='conv0',
                Convolution_num_output=num_output_final,
                Convolution_kernel_size=kernel,
                Convolution_pad=pad,
                Convolution_dilation=args.dilation,
                Convolution_weight_filler=filler,
                bottoms=['images'],
                tops=['RS_est_before_sigmoid']
            )
        )
        layers.append(SigmoidLayer(bottoms=['RS_est_before_sigmoid'],
                                   tops=['RS_est']))

    return layers


def create_convStaticSkipLayers(args, num_output_final):
    """Create network in barrista."""
    layers = []

    # if you want to use a Prelu, do the following below:
    # layers.append(PReLULayer())
    # or SigmoidLayer

    filler = design.PROTODETAIL.FillerParameter()
    filler.type = args.filler_type
    # filler.value = 1

    kernel = 2 * args.kernel_pad + 1
    pad = args.kernel_pad + (args.dilation - 1)
    num_output = 2**args.num_filters_log
    print("This net has skip layers and a sigmoid in the end!",
          "Using a convolutional network with ReLU units.",
          "It has", args.numLayers, "layers with",
          num_output, "filters each, then a conv layer with num_output",
          "filters (dependent on the RS_estimation mode)",
          "The kernels are of size", kernel, "with a padding of", pad)

    if args.numLayers >= 1:
        i = 0
        layers.append(
            ConvolutionLayer(
                name='conv{}'.format(i),
                bottoms=['images'],
                tops=['conv{}'.format(i)],
                Convolution_num_output=num_output,
                Convolution_kernel_size=kernel,
                Convolution_pad=pad,
                Convolution_dilation=args.dilation,
                Convolution_weight_filler=filler,
            )
        )
        if args.use_batch_normalization:
            layers.append(
                BatchNormLayer(
                    name='bn{}'.format(i),
                    bottoms=['conv{}'.format(i)],
                    tops=['conv{}'.format(i)],
                )
            )
        layers.append(
            ReLULayer(
                name='relu{}'.format(i),
                bottoms=['conv{}'.format(i)],
                tops=['conv{}'.format(i)],
            )
        )

        for i in range(1, args.numLayers):
            layers.append(
                ConvolutionLayer(
                    name='conv{}'.format(i),
                    bottoms=['conv{}'.format(i-1)],
                    tops=['conv{}'.format(i)],
                    Convolution_num_output=num_output,
                    Convolution_kernel_size=kernel,
                    Convolution_pad=pad,
                    Convolution_dilation=args.dilation,
                    Convolution_weight_filler=filler,
                )
            )
            if args.use_batch_normalization:
                layers.append(
                    BatchNormLayer(
                        name='bn{}'.format(i),
                        bottoms=['conv{}'.format(i)],
                        tops=['conv{}'.format(i)],
                    )
                )
            layers.append(
                ReLULayer(
                    name='relu{}'.format(i),
                    bottoms=['conv{}'.format(i)],
                    tops=['conv{}'.format(i)],
                )
            )

        layers.append(
            ConcatLayer(
                name='concat_skip_layers',
                bottoms=['conv{}'.format(i) for i in range(args.numLayers)],
                tops=['concat_skip_layers'],
            )
        )
        layers.append(
            ConvolutionLayer(
                name='fuse_skip_layers',
                bottoms=['concat_skip_layers'],
                tops=['RS_est_before_sigmoid'],
                Convolution_num_output=num_output_final,
                Convolution_kernel_size=1,
                Convolution_pad=0,
                Convolution_weight_filler=filler,
            )
        )
        layers.append(
            SigmoidLayer(
                name='sigmoid_after_fusing',
                bottoms=['RS_est_before_sigmoid'],
                tops=['RS_est'],
            )
        )
    else:
        # catch dummy case of numLayers == 0
        layers.append(
            ConvolutionLayer(
                name='conv0',
                Convolution_num_output=num_output_final,
                Convolution_kernel_size=kernel,
                Convolution_pad=pad,
                Convolution_dilation=args.dilation,
                Convolution_weight_filler=filler,
                bottoms=['images'],
                tops=['RS_est_before_sigmoid']
            )
        )
        layers.append(
            SigmoidLayer(
                name='sigmoid_after_fusing',
                bottoms=['RS_est_before_sigmoid'],
                tops=['RS_est'],
            )
        )

    return layers


def create_cascadeSkipLayers(args, num_output_final):
    """Create network in barrista."""
    layers = []

    # if you want to use a Prelu, do the following below:
    # layers.append(PReLULayer())
    # or SigmoidLayer

    filler = design.PROTODETAIL.FillerParameter()
    filler.type = args.filler_type
    # filler.value = 1

    kernel = 2 * args.kernel_pad + 1
    pad = args.kernel_pad + (args.dilation - 1)
    num_output = 2**args.num_filters_log
    print("This net has skip layers and a sigmoid in the end!",
          "Using a convolutional network with ReLU units.",
          "It has", args.numLayers, "layers with",
          num_output, "filters each, then a conv layer with num_output",
          "filters (dependent on the RS_estimation mode)",
          "The kernels are of size", kernel, "with a padding of", pad)

    if args.numLayers >= 1:
        i = 0
        layers.append(
            ConvolutionLayer(
                name='conv{}_level0'.format(i),
                bottoms=['images'],
                tops=['conv{}_level0'.format(i)],
                Convolution_num_output=num_output,
                Convolution_kernel_size=kernel,
                Convolution_pad=pad,
                Convolution_dilation=args.dilation,
                Convolution_weight_filler=filler,
            )
        )
        if args.use_batch_normalization:
            layers.append(
                BatchNormLayer(
                    name='bn{}_level0'.format(i),
                    bottoms=['conv{}_level0'.format(i)],
                    tops=['conv{}_level0'.format(i)],
                )
            )
        layers.append(
            ReLULayer(
                name='relu{}_level0'.format(i),
                bottoms=['conv{}_level0'.format(i)],
                tops=['conv{}_level0'.format(i)],
            )
        )

        for i in range(1, args.numLayers):
            layers.append(
                ConvolutionLayer(
                    name='conv{}_level0'.format(i),
                    bottoms=['conv{}_level0'.format(i-1)],
                    tops=['conv{}_level0'.format(i)],
                    Convolution_num_output=num_output,
                    Convolution_kernel_size=kernel,
                    Convolution_pad=pad,
                    Convolution_dilation=args.dilation,
                    Convolution_weight_filler=filler,
                )
            )
            if args.use_batch_normalization:
                layers.append(
                    BatchNormLayer(
                        name='bn{}_level0'.format(i),
                        bottoms=['conv{}_level0'.format(i)],
                        tops=['conv{}_level0'.format(i)],
                    )
                )
            layers.append(
                ReLULayer(
                    name='relu{}_level0'.format(i),
                    bottoms=['conv{}_level0'.format(i)],
                    tops=['conv{}_level0'.format(i)],
                )
            )

        layers.append(
            ConcatLayer(
                name='concat_skip_layers_level0',
                bottoms=['conv{}_level0'.format(i)
                         for i in range(args.numLayers)],
                tops=['concat_skip_layers_level0'],
            )
        )
        layers.append(
            ConvolutionLayer(
                name='fuse_skip_layers_level0',
                bottoms=['concat_skip_layers_level0'],
                tops=['RS_est_before_sigmoid_level0'],
                Convolution_num_output=num_output_final,
                Convolution_kernel_size=1,
                Convolution_pad=0,
                Convolution_weight_filler=filler,
            )
        )
        layers.append(
            SigmoidLayer(
                name='sigmoid_after_fusing_level0',
                bottoms=['RS_est_before_sigmoid_level0'],
                tops=['RS_est_level0'],
            )
        )
    else:
        # catch dummy case of numLayers == 0
        layers.append(
            ConvolutionLayer(
                name='conv0_level0',
                Convolution_num_output=num_output_final,
                Convolution_kernel_size=kernel,
                Convolution_pad=pad,
                Convolution_dilation=args.dilation,
                Convolution_weight_filler=filler,
                bottoms=['images'],
                tops=['RS_est_before_sigmoid_level0']
            )
        )
        layers.append(
            SigmoidLayer(
                name='sigmoid_after_fusing_level0',
                bottoms=['RS_est_before_sigmoid_level0'],
                tops=['RS_est_level0'],
            )
        )

    # recover reflectance and shading:
    layers.append(
        PythonLayer(
            Python_module='recover_reflectance_shading_layer',
            Python_layer='RecoverReflectanceShadingLayer',
            Python_param_str=args.RS_est_mode,
            name='recover_reflectance_shading_inner_level0',
            bottoms=['RS_est_level0',
                     'images'],
            tops=['reflectance_level0',
                  'shading_level0']
        )
    )
    # add loss layers for inner levels
    # compute whdr hinge loss
    bottoms = ['reflectance_level0', args.comparisonsType]
    if args.dataset == 'sintel':
        bottoms.append('albedos')
    layers.append(
        PythonLayer(
            Python_module='whdr_hinge_loss_layer',
            Python_layer='WhdrHingeLossLayer',
            Python_param_str=(args.whdr_delta_margin_ratio_dense),
            name='loss_whdr_hinge_level0',
            bottoms=bottoms,
            tops=['loss_whdr_hinge_level0'],
            loss_weights=[args.loss_scale_whdr],
            include_stages=['fit']
        )
    )
    # compute 'real' WHDR as 'accuracy' for evaluation
    bottoms = ['reflectance_level0', 'comparisons']
    if args.dataset == 'sintel':
        bottoms.append('albedos')
    layers.append(PythonLayer(Python_module='whdr_layer',
                              Python_layer='WhdrLayer',
                              Python_param_str="0.1",
                              name='whdr_original_level0',
                              bottoms=bottoms,
                              tops=['whdr_original_level0'],
                              # do not account as loss layer:
                              loss_weights=[0],
                              include_stages=['fit']))

    # define what to pass on to the next level
    level1_input = 'reflectance_level0'

    # add concatenation to input into next level
    # layers.append(
    #     ConcatLayer(
    #         name='concat_output_level0_to_input_level1',
    #         bottoms=['images', 'reflectance_level0', 'shading_level0'],
    #         tops=['concat_output_level0_to_input_level1'],
    #     )
    # )
    # level1_input = 'concat_output_level0_to_input_level1'

    if args.numLayers >= 1:
        i = 0
        layers.append(
            ConvolutionLayer(
                name='conv{}_level1'.format(i),
                bottoms=[level1_input],
                tops=['conv{}_level1'.format(i)],
                Convolution_num_output=num_output,
                Convolution_kernel_size=kernel,
                Convolution_pad=pad,
                Convolution_dilation=args.dilation,
                Convolution_weight_filler=filler,
            )
        )
        if args.use_batch_normalization:
            layers.append(
                BatchNormLayer(
                    name='bn{}_level1'.format(i),
                    bottoms=['conv{}_level1'.format(i)],
                    tops=['conv{}_level1'.format(i)],
                )
            )
        layers.append(
            ReLULayer(
                name='relu{}_level1'.format(i),
                bottoms=['conv{}_level1'.format(i)],
                tops=['conv{}_level1'.format(i)],
            )
        )

        for i in range(1, args.numLayers):
            layers.append(
                ConvolutionLayer(
                    name='conv{}_level1'.format(i),
                    bottoms=['conv{}_level1'.format(i-1)],
                    tops=['conv{}_level1'.format(i)],
                    Convolution_num_output=num_output,
                    Convolution_kernel_size=kernel,
                    Convolution_pad=pad,
                    Convolution_dilation=args.dilation,
                    Convolution_weight_filler=filler,
                )
            )
            if args.use_batch_normalization:
                layers.append(
                    BatchNormLayer(
                        name='bn{}_level1'.format(i),
                        bottoms=['conv{}_level1'.format(i)],
                        tops=['conv{}_level1'.format(i)],
                    )
                )
            layers.append(
                ReLULayer(
                    name='relu{}_level1'.format(i),
                    bottoms=['conv{}_level1'.format(i)],
                    tops=['conv{}_level1'.format(i)],
                )
            )

        layers.append(
            ConcatLayer(
                name='concat_skip_layers_level1',
                bottoms=['conv{}_level1'.format(i)
                         for i in range(args.numLayers)],
                tops=['concat_skip_layers_level1'],
            )
        )
        layers.append(
            ConvolutionLayer(
                name='fuse_skip_layers_level1',
                bottoms=['concat_skip_layers_level1'],
                tops=['RS_est_before_sigmoid_level1'],
                Convolution_num_output=num_output_final,
                Convolution_kernel_size=1,
                Convolution_pad=0,
                Convolution_weight_filler=filler,
            )
        )
        layers.append(
            SigmoidLayer(
                name='sigmoid_after_fusing_level1',
                bottoms=['RS_est_before_sigmoid_level1'],
                tops=['RS_est'],
            )
        )
    else:
        # catch dummy case of numLayers == 0
        layers.append(
            ConvolutionLayer(
                name='conv0_level1',
                Convolution_num_output=num_output_final,
                Convolution_kernel_size=kernel,
                Convolution_pad=pad,
                Convolution_dilation=args.dilation,
                Convolution_weight_filler=filler,
                bottoms=[level1_input],
                tops=['RS_est_before_sigmoid_level1']
            )
        )
        layers.append(
            SigmoidLayer(
                name='sigmoid_after_fusing_level1',
                bottoms=['RS_est_before_sigmoid_level1'],
                tops=['RS_est'],
            )
        )
        # last output needs to be 'RS_est' (without a level description)
    return layers


def create_convIncreasing(args, num_output_final):
    """Create network in barrista."""
    layers = []

    do_batch_normalization = False

    filler = design.PROTODETAIL.FillerParameter()
    filler.type = args.filler_type
    # filler.value = 1

    kernel = 2 * args.kernel_pad + 1
    pad = args.kernel_pad

    if args.numLayers >= 1:
        num_output = 2**args.num_filters_log
        num_outputs = [num_output]

        layers.append(
            ConvolutionLayer(
                Convolution_num_output=num_output,
                Convolution_kernel_size=kernel,
                Convolution_pad=pad,
                Convolution_weight_filler=filler,
                bottoms=['images']
            )
        )
        if do_batch_normalization:
            layers.append(BatchNormLayer())
        layers.append(ReLULayer())

        for i in range(1, args.numLayers):
            num_output *= 2
            num_outputs.append(num_output)
            layers.append(
                ConvolutionLayer(
                    Convolution_num_output=num_output,
                    Convolution_kernel_size=kernel,
                    Convolution_pad=pad,
                    Convolution_weight_filler=filler
                )
            )
            if do_batch_normalization:
                layers.append(BatchNormLayer())
            layers.append(ReLULayer())

        num_outputs.append(num_output_final)
        layers.append(
            ConvolutionLayer(
                Convolution_num_output=num_output_final,
                Convolution_kernel_size=1,
                Convolution_pad=0,
                Convolution_weight_filler=filler,
                tops=['RS_est']
            )
        )
    else:
        num_outputs = [num_output_final]
        layers.append(
            ConvolutionLayer(
                Convolution_num_output=num_output_final,
                Convolution_kernel_size=kernel,
                Convolution_pad=pad,
                Convolution_weight_filler=filler,
                bottoms=['images'],
                tops=['RS_est']
            )
        )

    print("Using a convolutional network with ReLU units where num_output.",
          "increases. It has", args.numLayers, "layers with",
          num_outputs, "filters, then a conv layer with num_output filters",
          "dependent on RS_est mode",
          "The kernels are of size", kernel, "with a padding of", pad)

    return layers
