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
Full pipeline for working with the WHDR CNN to train IIW.

Image convention is always a linear RGB image with shape
channels x height x width in the range 0 - 1.
"""
from __future__ import print_function, division

import os
import sys
import timeit
import platform
import datetime
import json
import logging
import traceback
# import pymongo
import cv2
import scipy.misc
import numpy as np

from tqdm import tqdm, trange

from data_handling import getData as _getData
from data_handling import rgb_to_srgb, srgb_to_rgb

# should be found since imported from parent
import caffe
import barrista

from networks import create_network
from extend_monitoring import (CombineLosses,
                               WHDRProgressIndicator,
                               RunningAverage,
                               CheckpointerIncludingRename,
                               )
from barrista.monitoring import (ProgressIndicator,
                                 ResultExtractor,
                                 )
from barrista import solver as _solver

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

# # compute WHDR with Bell's code
iiw_path = os.path.join(os.path.expanduser('~'),
                        'Datasets',
                        'intrinsic_images_in_the_wild',
                        'iiw-dataset')
sys.path.insert(0, iiw_path)
from whdr import compute_whdr  # noqa

# the values for which we evaluate
DELTA = 0.1
eps = np.finfo(np.float32).eps  # approx 1e-7 for threshold


def get_description(args):
    """Provide a descriptive string for the current setup in the experiment."""
    net_params = (args.networkType + '_' +
                  'n' + str(args.numLayers) + '_' +
                  'f' + str(2**args.num_filters_log) + '_' +
                  'k' + str(2*args.kernel_pad + 1) + '_' +
                  'd' + str(args.dilation) + '_' +
                  'bn' + str(args.use_batch_normalization) + '_' +
                  args.RS_est_mode + '_' +
                  'wdm' + args.whdr_delta_margin_ratio_dense
                  )

    def loss_format(l):
        if l:
            # form = "{:06.3f},"  # decimal
            form = "{:.1E},"  # scientific
            formatted = form.format(l)
            if formatted != form.format(0):  # has significant numbers
                return formatted
            else:
                # return "{:.1E},".format(l)  # show in scientific notation
                return str(l)  # show in regular python notation
        else:
            return "0,"

    losses = ("loss[" +
              "w" + loss_format(args.loss_scale_whdr) +
              "l" + loss_format(args.loss_scale_lambert)
              )
    losses = losses[:-1] + "]"  # remove the last "," and close the parantheses

    data_params = "h" + str(args.height) + "w" + str(args.width) + args.dataset

    description = (net_params + '_' +
                   losses + '_' +
                   args.solverType + str(args.base_lr) + '_' +
                   args.comparisonsType + '_' +
                   data_params)
    return net_params, description


message_was_already_shown = False
def write_to_database(collection_name, args):
    """
    Write entry to database.

    Add the entry containing of the parameters in 'args' into the database
    collection 'collection_name'.
    """
    # client = pymongo.MongoClient('10.34.27.85')
    # db = client['barrista_results']
    # collection = db[collection_name]
    # if hasattr(args, '_id'):
    #     # print("Deleting args._id", args._id)
    #     del args._id
    # post_id = collection.insert_one(vars(args))
    # client.close()
    # return post_id
    print("Please set up 'write_to_database' if you want to use it!")


def fit_predict_net(args, results_dir):
    """Train or test the network."""
    caffe.set_mode_gpu()  # run on GPU
    # collection_name = 'cluster'  # not used in code release

    net_params, description = get_description(args)

    snapshot_dir = os.path.join(results_dir, 'snapshots')
    draw_net_filename = os.path.join(results_dir, 'networks',
                                     net_params + '.png')

    additional_info = '_{}_{}_{}'.format(args.height, args.width,
                                         args.sRGB_linear)

    def getData(description):
        """Wrapper around the external getData using the given args."""
        return _getData(args.dataset,
                        description + additional_info,
                        args.comparisonsType)

    # progress indicator
    # progress = WHDRProgressIndicator(1, 1, 1)
    # progress = WHDRProgressIndicator(50, 100, 0.01)
    # the scales are just for human display, not for loss computation!
    progress = WHDRProgressIndicator(args.loss_scale_whdr,
                                     args.loss_scale_boundaries01,
                                     args.loss_scale_lambert)

    flags_fit = ['fit', 'f', 'train']
    flags_predict = ['predict', 'p', 'test', 'val']

    if args.stage in (flags_fit + flags_predict):
        print("Descriptive string:", description)
        # create the network of type networkType

        # print("Create network.")
        net = create_network(args, draw_net_filename)
        # print("Network created.")

        # if len(args.iterations) > 1:
        #     print("Use only one number of iterations for the maximum number"
        #           "of iterations to train. Will use the one provided last")
        # iterations = args.iterations[-1]

        iterations = args.iterations
        if args.iterations is None:
            if args.stage in flags_fit:
                raise Exception("Number of iterations was not set!")
            else:
                iterations = 1  # dummy

        # get data
        if not args.test:
            if args.stage in flags_fit:
                # no test mode, stage fit
                X = getData('trainValTest_train')
                # print("CHANGE AGAIN, READING DUMMY DATA!!!!!!!!!!!!!!!!!!!!")
                # X = getData('dummy_train')
                # print("CHANGE AGAIN, READING DUMMY DATA!!!!!!!!!!!!!!!!!!!!")
            # no test mode, stage predict and fit
            X_val = getData('trainValTest_val')
        else:  # --test=1
            if args.stage in flags_fit:
                # test mode, stage fit

                # train = trainValTest_train + trainValTest_val
                # X = getData('train')

                # instead use a most of the previous validation set for
                # training and leave only small validation
                X = getData('bigTrainMiniValTest_train')
                # as described above, bigger train set, small validation
                X_val = getData('bigTrainMiniValTest_val')
            elif args.stage in flags_predict:
                # test mode, stage predict
                # test data is only used when in --test=1 --stage=predict
                X_val = getData('trainValTest_test')


        # Define result extractor
        log_results = ['whdr_original',
                       'loss_whdr_hinge',
                       'whdr_original_level0',
                       'loss_whdr_hinge_level0',
                       'loss_boundaries01',
                       'loss_boundaries_reflectance',
                       'loss_boundaries_shading',
                       'loss_lambert']
        result_extractors = []
        for log_blob in log_results:
            result_extractors.append(ResultExtractor(log_blob, log_blob))
        combineLosses = CombineLosses(args.loss_scale_whdr,
                                      args.loss_scale_lambert)
        result_extractors.append(combineLosses)
        log_results.append('loss_combined')  # to tell the JSONLogger

        # Snapshots
        # snapshots_prefix = os.path.join(snapshot_dir, description + '_')
        snapshots_prefix = os.path.join(snapshot_dir, description)
        # print("snapshots prefix:", snapshots_prefix)
        checkpoint_interval = min(args.checkpoint_interval, iterations)
        print("Checkpointing every", args.checkpoint_interval, "iterations.")

        # Run the training.
        solver = _get_solver(args, snapshots_prefix)

        # print('name_prefix', snapshots_prefix,
        #       'iterations', checkpoint_interval)
        checkptr = CheckpointerIncludingRename(name_prefix=snapshots_prefix,
                                               iterations=checkpoint_interval)
        # json logs
        # filename will be prefixed with barrista_ and .json will be appended
        # json_log = JSONLogger(os.path.join(results_dir, 'logs'),
        #                       description + '_' + str(iterations),
        #                       {'train': logging})

        train_cb = list(result_extractors)
        # train_cb.append(json_log)  # for now do not use the json logger
        train_cb.append(checkptr)
        # test_cb = list(result_extractors)
        # test_cb.extend([json_log])

        train_cb.append(progress)
        # test_cb.append(progress)

        # print("Log the following blobs:")
        # for re in result_extractors:
        #     print('key:', re._cbparam_key, 'and layer_name:', re._layer_name)

        if args.stage in flags_fit:
            running_average = RunningAverage(X['images'].shape[0],
                                             args.batch_size)
            train_cb.append(running_average)

            start_train = timeit.default_timer()
            # print("Testing every", args.test_interval, "iterations.")
            print("Starting the training for", iterations, "iterations.")
            # flush the stdout (write content to file in cluster) for debugging
            sys.stdout.flush()

            if args.startOver:
                if args.predictCaffemodel:
                    print("Load initial weights from:", args.predictCaffemodel)
                    net.load_blobs_from(args.predictCaffemodel)

                net.fit(iterations,  # number of iterations
                        solver,
                        X,
                        # test_interval=args.test_interval,
                        # X_val=X_val,
                        train_callbacks=train_cb,
                        # test_callbacks=test_cb,
                        allow_test_phase_for_train=True,
                        )
            end_train = timeit.default_timer()
            training_time = end_train-start_train
            print("Total training time on node", platform.node(),
                  "is", training_time)

            # in the end evaluate the final model
            curr_iter = iterations
            cm = '_barrista_iter_{}.caffemodel'.format(curr_iter)
            caffemodel = description + cm
            print("Now predict data from val and evaluate the WHDR on it.")
            score = _predictCaffemodel(X_val, net, caffemodel, results_dir,
                                       args)

            args.score = score
            args.datetime = datetime.datetime.now()
            args.training_time = training_time
            # write everything into database
            # write_to_database(collection_name, args)

            # and evaluate all intermediate models
            print("Test all intermediate caffemodels.")
            json_val = []
            json_train = []
            scores = []
            for i in range(checkpoint_interval,  # start from first trained
                           iterations+1,
                           checkpoint_interval):
                curr_iter = i
                cm = '_barrista_iter_{}.caffemodel'.format(curr_iter)
                caffemodel = description + cm
                # save progression of val
                val_score = _predictCaffemodel(X_val,
                                               net,
                                               caffemodel,
                                               results_dir,
                                               args)
                json_val.append({"NumIters": curr_iter,
                                 "WHDR": val_score
                                 })
                # # if you also want to see the progression of train
                # # (takes longer!)
                # train_score = _predictCaffemodel(X,
                #                                  net,
                #                                  caffemodel,
                #                                  results_dir,
                #                                  args)
                # json_train.append({"NumIters": curr_iter,
                #                    "WHDR": train_score
                #                    })

                # # also insert into database
                # args.iterations = curr_iter
                # args.score = val_score
                # args.datetime = datetime.datetime.now()
                # write_to_database(collection_name, args)

                scores.append(val_score)

                print("Ran iteration", i, "of", iterations,
                      "with validation score", val_score)
                sys.stdout.flush()

            filename = os.path.join(results_dir, 'progressions',
                                    "barrista_" + description + ".json")
            with open(filename, 'w') as outfile:
                json.dump({"test": json_val, "train": json_train}, outfile)

            print("Final score in % (the best one):")
            # print(score)
            print(min(scores))

        if args.predictCaffemodel and args.stage in flags_predict:
            # X_val = getData('dummy_val')  # should already be loaded

            # parse parameters for network from filename
            caffemodel_abs = args.predictCaffemodel
            caffemodel_rel = os.path.basename(args.predictCaffemodel)
            desc_split = caffemodel_rel.split('_')
            # print("split description:", desc_split)
            args.networkType = desc_split[0]
            args.numLayers = int(desc_split[1][1:])
            args.num_filters_log = int(np.log2(int(desc_split[2][1:])))
            args.kernel_pad = int((int(desc_split[3][1:]) - 1) / 2)
            args.RS_est_mode = desc_split[6]
            args.whdr_delta_margin_ratio_dense = (desc_split[7][3:] + '_' +
                                                  desc_split[8] + '_' +
                                                  desc_split[9] + '_' +
                                                  desc_split[10])
            args.iterations = int(desc_split[-1][:-11])
            print("Inferred parameters:",
                  "\nnetworkType:", args.networkType,
                  "\nnumLayers:", args.numLayers,
                  "\nkernel_pad:", args.kernel_pad,
                  "\nnum_filters_log", args.num_filters_log,
                  "\nRS_est_mode:", args.RS_est_mode,
                  "\niterations:", args.iterations,
                  "\nwhdr_delta_mar_r_d:", args.whdr_delta_margin_ratio_dense,
                  )

            scores = []

            if args.decompose:
                print("Decompose input")

                files_to_decompose = []
                for entry in args.decompose:
                    if os.path.isfile(entry):
                        files_to_decompose.append(entry)
                    elif os.path.isdir(entry):
                        # files_to_decompose.extend(os.listdir(entry))
                        for f in os.listdir(entry):
                            files_to_decompose.append(os.path.join(entry, f))
                    else:
                        print(entry, "is neither a file nor folder")

                for i, img_vid in enumerate(tqdm(files_to_decompose)):
                    try:
                        if is_image(img_vid):
                            # old version, can be removed (does resizing to
                            # 256 x 256 and then upscales again)
                            # _decompose_images(img_vid, caffemodel_abs,
                            #                   results_dir, args)
                            # new version (does decomp.in full resolution)
                            _decompose_single_image_in_full_size(img_vid,
                                                                 caffemodel_abs,
                                                                 results_dir,
                                                                 args)
                        elif is_movie(img_vid):
                            _decompose_movie(img_vid, caffemodel_abs,
                                             results_dir, args)
                        elif is_numpy(img_vid):
                            _decompose_numpy(img_vid, caffemodel_abs,
                                             results_dir, args)
                        else:
                            print("\nFile", img_vid,
                                  "neither recognized as image, nor movie")
                    except:
                        print("Decomposing file",
                              img_vid,
                              "was not possible")
                        traceback.print_exc()
                return
    else:
        msg = "stage '{}' is currently not implemented!".format(args.stage)
        raise Exception(msg)


def percent(num):
    """Format number as per cent with 2 decimal places."""
    return "{:.2f}%".format(num * 100)


def _get_solver(args, snapshot_prefix):
    params = {'base_lr': args.base_lr,
              'random_seed': args.random_seed,
              'snapshot_prefix': snapshot_prefix}
    if args.solverType in ['SGD', 'sgd']:
        solver = _solver.SGDSolver(**params)
    elif args.solverType in ['ADAM', 'Adam', 'adam']:
        solver = _solver.AdamSolver(**params)
    else:
        raise Exception("solverType not known", args.solverType)
    # print('Using', args.solverType,
    #       'with learning rate', args.base_lr,
    #       'as solver.')
    return solver


def _get_highest_iteration(snapshot_dir, description):
    # do not use fnmatch or glob because of [] in description
    # find the files that start with the description
    files = [f for f in os.listdir(snapshot_dir)
             if f[:len(description)] == description]
    latest_caffe_iter = 0
    for f in files:
        # extract the iteration
        latest_caffe_iter = max(latest_caffe_iter,
                                int(f[len(description)+1:-11]))
    return latest_caffe_iter


def _load_highest_iteration(snapshot_dir, description, net):
    latest_caffe_iter = _get_highest_iteration(snapshot_dir, description)
    if latest_caffe_iter > 0:
        filename = (description + '_' + str(latest_caffe_iter) +
                    '.caffemodel')
        caffemodel = os.path.join(snapshot_dir, filename)
        net.load_blobs_from(caffemodel)
        print("Continuing from iteration", latest_caffe_iter,
              "with file", caffemodel)
    else:
        print("No previously trained net found, starting from scratch.")
    return latest_caffe_iter


def _predictCaffemodel(X_val, net, caffemodel_rel, results_dir,
                       args):
    description = caffemodel_rel[:-11]  # description including iteration
    caffemodel_abs = os.path.join(results_dir, 'snapshots', caffemodel_rel)
    return _predictCaffemodel_abs(X_val, net, caffemodel_abs, results_dir,
                                  description, args)


def _predictCaffemodel_abs(X_val, net, caffemodel_abs, results_dir,
                           description, args):
    # before_loading = net.params['_layer_0'][0].data[0, 0, :, :].copy()

    net.load_blobs_from(caffemodel_abs)
    # after_loading = net.params['_layer_0'][0].data[0, 0, :, :].copy()

    num_images = X_val['images'].shape[0]
    description += "_imgs{}".format(num_images)

    score_filename = os.path.join(results_dir, 'scores',
                                  description + '.txt')

    if os.path.isfile(score_filename):
        # If the caffemodel was already evaluated, read its score file and
        # return its content.
        with open(score_filename, 'r') as scores_file:
            result = float(scores_file.readline())
            if result < 100:
                return result
            # a default 100 should not have been written to a score file,
            # but still:
            # otherwise continue getting the score from the caffemodel


    out_blob_names = ['RS_est',
                      'reflectance',
                      'shading']

    if args.networkType == 'cascadeSkipLayers':
        out_blob_names.append('reflectance_level0')

    start_predicting = timeit.default_timer()
    try:
        input_sequence = {'images': X_val['images']}
        results = net.predict(input_sequence,
                              test_callbacks=[ProgressIndicator()],
                              out_blob_names=out_blob_names)
    except:
        traceback.print_exc()
        print("Prediction was not possible, returning 100 as default!")
        return 100
    stop_predicting = timeit.default_timer()

    prediction_time = stop_predicting-start_predicting
    print("Predicting", num_images, "images took",
          prediction_time, "seconds, i.e., ",
          prediction_time/num_images, "per image and",
          num_images/prediction_time, "images per second.")

    # write frame rate to file
    framerate_filename = os.path.join(results_dir, 'framerates',
                                      description + '.txt')
    with open(framerate_filename, 'w') as framerate_file:
        framerate_file.write(str(num_images/prediction_time))

    # RS_estimations = results['RS_est']
    reflectances = results['reflectance']
    # shadings = results['shading']

    # these are now LISTS of 3D images, not 4D blobs! So one can access e.g.
    # reflectances[b][c, h, w]

    # for debug, save first output reflectance image
    # _save_img('debug.png', reflectances[0])

    # evaluate WHDR on whole validation set
    whdrs = [_whdr_bell(_transp(reflectances[b]), _orig_filename(X_val, b))
             for b in range(num_images)]

    meanWHDR = np.mean(whdrs)
    score = meanWHDR * 100  # score is WHDR in %

    print('WHDR on learned reflectance for caffemodel:', description)
    print('WHDRs:',
          '\t min', percent(min(whdrs)),
          '\t max', percent(max(whdrs)),
          '\t median', percent(np.median(whdrs)),
          '\t mean', percent(meanWHDR),
          )

    # print("Write score {} to file {}".format(score, score_filename))
    with open(score_filename, 'w') as score_file:
        score_file.write(str(score))

    return score  # return the score in %


def is_image(filename):
    """Check if a file is an image (by extension)."""
    extension = os.path.splitext(filename)[1][1:].strip().lower()
    return extension in ['jpg', 'png', 'ppm', 'tiff']


def is_movie(filename):
    """Check if a file is a movie (by extension)."""
    extension = os.path.splitext(filename)[1][1:].strip().lower()
    return extension in ['mp4', 'avi']


def is_numpy(filename):
    """Check if a file is a numpy file (by extension)."""
    extension = os.path.splitext(filename)[1][1:].strip().lower()
    return extension in ['npz']


def _img_to_blob(img, args, resize=True):
    if resize:
        img = scipy.misc.imresize(img, (args.height, args.width))
    img = np.transpose(img, (2, 0, 1))
    img = img / 255
    return img


def _blob_to_img(img, height=None, width=None):
    # only used in movie now, but check, since convention changed!
    img = (img * 255).astype(int)
    img = np.transpose(img, (1, 2, 0))
    if height is not None and width is not None:
        img = scipy.misc.imresize(img, (height, width))
    return img


def _numpy2cv(frame):
    rgb = _transp(frame)
    srgb = rgb_to_srgb(rgb)
    # TODO: should we normalize here?
    # rgb_order = (srgb/np.max(srgb) * 255).astype('u1')

    # without normalizing, but thresholding
    # (without gives overflow artifacts)
    rgb_order = (threshold_image(srgb) * 255).astype('u1')
    # print("values:", np.min(rgb_order), np.max(rgb_order))
    bgr_order = cv2.cvtColor(rgb_order, cv2.COLOR_RGB2BGR)
    return bgr_order


def threshold_image(img):
    """Threshold an input ndarray to the range 0-1."""
    # return np.minimum(np.maximum(img, 0), 1)
    return np.clip(img, 0, 1)


def _transp(img):  # change from c, y, x to y, x, c
    return np.transpose(img, (1, 2, 0))


def _resize(img, height, width):
    return cv2.resize(img, dsize=(width, height))


def _color(img):
    return np.tile(img, (3, 1, 1))


def _read_img(full_path):
    print("Reading image from:", full_path)
    img = cv2.imread(full_path)
    img = img[:, :, ::-1]  # BGR -> RGB
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(float) / 255.0
    img = srgb_to_rgb(img)
    # now img is a linear RGB image with shape chw and range 0-1
    # print("After reading image:", img.shape, np.min(img), np.max(img))
    return img


def _save_img(full_path, img_in,
              height=None, width=None,
              scale2Max=False, convert2sRGB=False):
    img = img_in.copy()
    # if img.shape[0] == 1:
    #     img = _color(img)  # colorize grayscale
    img = np.transpose(img, (1, 2, 0))  # change to regular cv2 image order
    img = img[:, :, ::-1]
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # expects correct dtype
    if height is not None and width is not None:
        img = _resize(img, height, width)
    if scale2Max:
        # this should keep WHDR intact (ratios stay the same)
        # and keeps Lambertian assumption besides a global scaling factor
        img /= np.max(img)
        # lower shift to 0 is not allowed (it is not a scaling!)
    if convert2sRGB:
        img = rgb_to_srgb(img)  # make it an sRGB image
    # print("Before writing image:", img.shape, np.min(img), np.max(img))
    # explicit casting to np.uint8 makes it slightly worse
    # probably cv2 does correct rounding instead of floor as with astype
    cv2.imwrite(full_path, img * 255)


def _decompose_4d_blob(blob, net):
    results = net.predict(blob, out_blob_names=['reflectance',
                                                'shading',
                                                'RS_est',
                                                ])
    reflectance = np.stack(results['reflectance'])
    shading = np.stack(results['shading'])
    RS_est = np.stack(results['RS_est'])
    return reflectance, shading, RS_est


def _decompose_3d_blob(blob, net):
    results = net.predict([blob], out_blob_names=['reflectance',
                                                  'shading',
                                                  'RS_est',
                                                  ])
    reflectance = results['reflectance'][0]
    shading = results['shading'][0]
    RS_est = results['RS_est'][0]
    return reflectance, shading, RS_est


def _decompose_numpy(img_vid, caffemodel_abs, results_dir, args):
    # read input numpy file. Expect input in 'images' to have shape:
    # (num_images, height, width, channels)
    with np.load(img_vid) as npzFile:
        images = npzFile['images']
    # print(images.shape)

    # convert to blob for caffe (assume once linear, and once sRGB)
    input_as_is = np.transpose(images/255, (0, 3, 1, 2))
    # print(input_as_is.shape)

    # set network size to image size
    args.height, args.width = input_as_is.shape[2:]
    net = create_network(args)
    net.load_blobs_from(caffemodel_abs)

    # get decomposition of blob (once assumed as linear, once as sRGB)
    # unchanged (when assuming input to be linear, output will be kept that)
    R_from_input, S_from_input, r_from_input = _decompose_4d_blob(input_as_is,
                                                                  net)
    R_from_input = np.transpose(R_from_input, (0, 2, 3, 1))
    S_from_input = np.transpose(S_from_input, (0, 2, 3, 1))
    r_from_input = np.transpose(r_from_input, (0, 2, 3, 1))
    # when converting input from sRGB, undo with R and S
    input_converted_to_linear = srgb_to_rgb(input_as_is)
    R, S, r = _decompose_4d_blob(input_converted_to_linear, net)
    R_back_to_sRGB = np.transpose(rgb_to_srgb(R), (0, 2, 3, 1))
    S_back_to_sRGB = np.transpose(rgb_to_srgb(S), (0, 2, 3, 1))
    r_back_to_sRGB = np.transpose(rgb_to_srgb(r), (0, 2, 3, 1))

    # save as numpy file back to where original filename was
    np.savez_compressed(img_vid[:-4] + '_decomposed.npz',
                        images=images,
                        R_back_to_sRGB=R_back_to_sRGB,
                        S_back_to_sRGB=S_back_to_sRGB,
                        r_back_to_sRGB=r_back_to_sRGB,
                        R_from_input=R_from_input,
                        S_from_input=S_from_input,
                        r_from_input=r_from_input,
                        )


def _decompose_single_image_in_full_size(img_vid, caffemodel_abs,
                                         results_dir, args):
    img = _read_img(img_vid)

    # set network size to image size
    args.height, args.width = img.shape[1:]
    net = create_network(args)
    net.load_blobs_from(caffemodel_abs)

    # get decomposition of blob
    reflectance, shading, RS_est = _decompose_3d_blob(img, net)

    # output name is input name (in another folder)
    orig_filename = os.path.basename(img_vid)[:-4]
    # in the case they all have the same name (e.g. in MIT intrinsic), prepend
    # the directory name:
    # dirname = os.path.basename(os.path.dirname(img_vid))
    # orig_filename = dirname + '_' + os.path.basename(img_vid)[:-4]

    # png actually does not look that much better than jpg
    # but is way bigger file size (since lossless compression).
    # IIW decompositions are all given in png

    img_format = '.png'
    # img_format = '.jpg'

    # decompose without scaling in linear
    full_path = os.path.join(results_dir,
                             'decompositions_linear',
                             orig_filename + '-r' + img_format)
    _save_img(full_path, reflectance, scale2Max=False, convert2sRGB=False)
    full_path = os.path.join(results_dir,
                             'decompositions_linear',
                             orig_filename + '-s' + img_format)
    _save_img(full_path, shading, scale2Max=False, convert2sRGB=False)
    full_path = os.path.join(results_dir,
                             'decompositions_linear',
                             orig_filename + '-RS_est' + img_format)
    _save_img(full_path, RS_est, scale2Max=False, convert2sRGB=False)

    # and in sRGB
    full_path = os.path.join(results_dir,
                             'decompositions_sRGB',
                             orig_filename + '-r' + img_format)
    _save_img(full_path, reflectance, scale2Max=False, convert2sRGB=True)
    full_path = os.path.join(results_dir,
                             'decompositions_sRGB',
                             orig_filename + '-s' + img_format)
    _save_img(full_path, shading, scale2Max=False, convert2sRGB=True)
    full_path = os.path.join(results_dir,
                             'decompositions_sRGB',
                             orig_filename + '-RS_est' + img_format)
    _save_img(full_path, RS_est, scale2Max=False, convert2sRGB=True)

    # # write the combined image
    # combined = np.concatenate([img,
    #                            _color(RS_est),
    #                            reflectance,
    #                            shading,
    #                            ], axis=2)
    # full_path = os.path.join(results_dir,
    #                          'decompositions',
    #                          orig_filename + '-combined' + img_format)
    # # _save_img(combined, full_path, orig_height, 4*orig_width)
    # scipy.misc.imsave(full_path, combined)

    # # do normalization of maximum to 1 in linear
    # full_path = os.path.join(results_dir,
    #                          'decompositions_normalized_linear',
    #                          orig_filename + '-r' + img_format)
    # _save_img(full_path, reflectance, scale2Max=True, convert2sRGB=False)
    # full_path = os.path.join(results_dir,
    #                          'decompositions_normalized_linear',
    #                          orig_filename + '-s' + img_format)
    # _save_img(full_path, shading, scale2Max=True, convert2sRGB=False)
    # full_path = os.path.join(results_dir,
    #                          'decompositions_normalized_linear',
    #                          orig_filename + '-RS_est' + img_format)
    # _save_img(full_path, RS_est, scale2Max=True, convert2sRGB=False)
    # # and with sRGB
    # full_path = os.path.join(results_dir,
    #                          'decompositions_normalized_sRGB',
    #                          orig_filename + '-r' + img_format)
    # _save_img(full_path, reflectance, scale2Max=True, convert2sRGB=True)
    # full_path = os.path.join(results_dir,
    #                          'decompositions_normalized_sRGB',
    #                          orig_filename + '-s' + img_format)
    # _save_img(full_path, shading, scale2Max=True, convert2sRGB=True)
    # full_path = os.path.join(results_dir,
    #                          'decompositions_normalized_sRGB',
    #                          orig_filename + '-RS_est' + img_format)
    # _save_img(full_path, RS_est, scale2Max=True, convert2sRGB=True)


def _decompose_multiple_frames(img, net, args):
    # this assumes that all frames have the same size and are input
    # as a list of images

    # remember input size for later
    height, width = img[0].shape[:2]
    blob = np.empty((len(img), 3, args.height, args.width))
    for b in range(len(img)):
        # convert 0-255 int hwc image to 0-1 chw blob of CNN size
        blob[b, :, :, :] = srgb_to_rgb(_img_to_blob(img, args))
    # get decomposition of blob
    reflectance, shading, RS_est = _decompose_4d_blob(blob, net)[:2]
    # go back to image format and resize
    reflectances = []
    shadings = []
    for b in range(len(img)):
        # check _blob_to_img, change drastically now!!
        reflectances.append(_blob_to_img(reflectance[b, :, :, :],
                                         height, width))
        shadings.append(_blob_to_img(shading[b, :, :, :], height, width))
    return reflectances, shadings


def load_movie(filename, args):
    """Load movie with open cv as blob."""
    # start_load = timeit.default_timer()
    cap = cv2.VideoCapture(filename)
    cap.open(filename)
    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    # print("Video statistics: length, width, height, fps",
    #       length, width, height, fps)
    frames = []
    print("Loading movie", filename)

    for i in trange(length):  # trange = tqdm(range)
        if cap.isOpened():
            ret, frame = cap.read()
            # print("Reading frame successfull:", ret)

            # do not resize movie
            bgr = frame
            # # To shrink an image, it will generally look best with
            # # CV_INTER_AREA interpolation, whereas to enlarge an image, it will
            # # generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR
            # # (faster but still looks OK).
            # bgr = cv2.resize(frame, (args.height, args.width),
            #                  interpolation=cv2.INTER_AREA)

            srgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            linear = srgb_to_rgb(srgb/255)
            # pdb.set_trace()
            frames.append(np.transpose(linear, (2, 0, 1)))
    # stop_load = timeit.default_timer()
    # print("Loading movie took {} seconds".format(stop_load - start_load))
    return np.array(frames), [width, height, fps]


def _resize_movie(frame, width, height):
    # do not resize movie
    return _numpy2cv(frame)

    # # To shrink an image, it will generally look best with
    # # CV_INTER_AREA interpolation, whereas to enlarge an image, it will
    # # generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR
    # # (faster but still looks OK).
    # inter = cv2.INTER_CUBIC  # slow
    # # inter = cv2.INTER_LINEAR  # faster
    # return cv2.resize(_numpy2cv(frame), (width, height), interpolation=inter)


def save_movie_combined(filename, image, reflectance, shading, stats):
    """Take a list of 3d blobs and turn into movie (concat img, refl, shad)."""
    start_save = timeit.default_timer()

    width, height, fps = stats

    codec = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')  # note the lower case
    # codec = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')
    # codec = cv2.cv.CV_FOURCC('A', 'V', 'C', '1')

    videoFile = cv2.VideoWriter()
    name = filename[:-4] + "-combined" + filename[-4:]
    print("Writing to file", name)
    videoFile.open(name, codec, fps, (3*width, height), True)

    length = image.shape[0]

    for i in trange(length):
        img = _resize_movie(image[i, :, :, :], width, height)
        refl = _resize_movie(reflectance[i], width, height)
        # global normalization of shading is done here
        shad = _resize_movie(shading[i], width, height)
        # shad = _resize_movie(shading[i]/np.max(shading), width, height)
        frame = np.concatenate([img, refl, shad], axis=1)
        videoFile.write(frame)

    # print("Finish writing video")
    videoFile.release()
    videoFile = None

    stop_save = timeit.default_timer()
    print("Saving movie took {} seconds".format(stop_save - start_save))


def save_movie_separate(filename, image, reflectance, shading, stats):
    """Take a list of 3d blobs and turn into movie (concat img, refl, shad)."""
    start_save = timeit.default_timer()

    width, height, fps = stats

    codec = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')  # note the lower case
    # codec = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')
    # codec = cv2.cv.CV_FOURCC('A', 'V', 'C', '1')

    # first the reflectance video
    videoFile = cv2.VideoWriter()
    name = filename[:-4] + "-r" + filename[-4:]
    videoFile.open(name, codec, fps, (width, height), True)

    length = image.shape[0]

    for i in trange(length):
        refl = _resize_movie(reflectance[i], width, height)
        videoFile.write(refl)

    videoFile.release()

    # now the shading video
    videoFile = cv2.VideoWriter()
    name = filename[:-4] + "-s" + filename[-4:]
    videoFile.open(name, codec, fps, (width, height), True)

    length = image.shape[0]

    for i in trange(length):
        # global normalization of shading is done here
        shad = _resize_movie(shading[i], width, height)
        # shad = _resize_movie(shading[i]/np.max(shading), width, height)
        videoFile.write(shad)

    # print("Finish writing video")
    videoFile.release()
    videoFile = None

    stop_save = timeit.default_timer()
    print("Saving movie took {} seconds".format(stop_save - start_save))


def save_movie_baseline(filename, image, stats):
    """Create from a 3d blob of frames a baseline reflectance and shading."""
    print("Creating baseline movie")
    eps = np.finfo(np.float32).eps  # approx 1e-7 for thresholding
    length = image.shape[0]

    baseline_filename = filename[:-4] + "-baseline_rgbMean" + filename[-4:]
    reflectance = []
    shading = []
    for i in trange(length):
        img = image[i, :, :, :]
        img_intensity = np.sum(img, axis=0, keepdims=True) / 3
        img_intensity = np.maximum(img_intensity, eps)
        reflectance.append(img/img_intensity)
        shading.append(np.tile(img_intensity, (3, 1, 1)))
    save_movie_combined(baseline_filename, image, reflectance, shading, stats)

    baseline_filename = filename[:-4] + "-baseline_rgbNorm" + filename[-4:]
    reflectance = []
    shading = []
    for i in trange(length):
        img = image[i, :, :, :]
        img_intensity = np.linalg.norm(img, axis=0)[np.newaxis, :, :]
        img_intensity = np.maximum(img_intensity, eps)
        reflectance.append(img/img_intensity)
        shading.append(np.tile(img_intensity, (3, 1, 1)))
    save_movie_combined(baseline_filename, image, reflectance, shading, stats)


def _decompose_movie(img_vid, caffemodel_abs, results_dir, args):
    images, stats = load_movie(img_vid, args)
    num_images = images.shape[0]
    orig_filename = os.path.basename(img_vid)[:-4]
    full_path = os.path.join(results_dir,
                             'decompositions_sRGB',
                             orig_filename + '.mp4')

    # generate baseline reflectance video
    save_movie_baseline(full_path, images, stats)

    # set network size to movie size
    args.height = stats[1]
    args.width = stats[0]
    net = create_network(args)
    net.load_blobs_from(caffemodel_abs)

    print("Decompose movie")
    start_predicting = timeit.default_timer()
    results = net.predict(images,
                          post_batch_callbacks=[ProgressIndicator()],
                          out_blob_names=['reflectance', 'shading'])
    stop_predicting = timeit.default_timer()

    prediction_time = stop_predicting-start_predicting
    print("Predicting", num_images, "frames took",
          prediction_time, "seconds, i.e., ",
          prediction_time/num_images, "per frame and",
          num_images/prediction_time, "fps.")

    reflectances = results['reflectance']
    shadings = results['shading']
    save_movie_combined(full_path, images, reflectances, shadings, stats)
    save_movie_separate(full_path, images, reflectances, shadings, stats)



def _orig_filename(X, b):
    return str(int(X['comparisons'][b, -1, 0, 1]))


def _whdr_bell(img, fileID):
    if (img.ndim != 3 or img.shape[0] != 256 or
            img.shape[1] != 256 or (img.shape[2] != 3 and img.shape[2] != 1)):
        print("Input image into _whdr_bell did not have expected shape:",
              "expected: (256, 256, 3), received:", img.shape,
              "Are you sure you know what you are doing?")
    filename = fileID + '.json'
    full_path = os.path.join(iiw_path, 'data', filename)
    return compute_whdr(img, json.load(open(full_path)))
