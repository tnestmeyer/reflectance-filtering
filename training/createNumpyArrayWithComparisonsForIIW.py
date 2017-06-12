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

"""Gets comparisons from json files and saves them to a file."""

from __future__ import print_function, division

# the next two lines are to avoid showing the Rocket Symbol in OS X, see
# http://leancrew.com/all-this/2014/01/stopping-the-python-rocketship-icon/
import matplotlib
matplotlib.use("Agg")  # noqa

import multiprocessing
import time

import timeit
import os
import sys
# import cv2
import scipy.misc
import simplejson as json
import numpy as np

from tqdm import tqdm

start = timeit.default_timer()

##############################################################################
RESOLUTIONS = [[256, 256]]
# RESOLUTIONS = [[340, 512]]
# RESOLUTIONS = [[256, 256], [192, 256]]
# ,[170, 256], [200, 256], [288, 288]]

NARIHIRA = True  # use the method of Narihira to split (80% train, 20% test)
# or in the case of train val and test: 70, 10, 20
SPLIT = 0.9  # define the ratio: SPLIT into train, 1-SPLIT into test (if not
# Narihira's method)

# CREATE = ['one']            # create dummy data with 1 and 2 images
# CREATE = ['dummy']          # create dummy data with 10 images
# CREATE = ['trainTest']      # create training and test set
CREATE = ['trainValTest']   # create a training, validation and test set
# CREATE = ['all']            # create all data in one file
# CREATE = ['allShuffled']    # create all data shuffled in one file

# CREATE = ['dummy', 'trainValTest', 'bigTrainMiniValTest']
# CREATE = ['one', 'dummy', 'trainTest', 'trainValTest', 'all', 'allShuffled']
# create all possibilities

DATA_FOLDER = os.path.join(os.path.expanduser('~'),
                           'iiw_additions',
                           'iiw-dataset',
                           'data')
IMAGE_EXTENSION = '.png'
SAVE_TO_FOLDER = os.path.join(os.path.expanduser('~'),
                              'LMDBs',
                              'new_iiw_for_code_release')
# for convenience append '/' to the path to use without os.path
SAVE_TO_FOLDER = SAVE_TO_FOLDER + '/'

# empirically found out
MAX_NUM_COMPARISONS = 1181
AUGMENT_DATA = False
MAX_NUM_AUGMENTED = 60049  # 89546  # when also accepting A=B and B=A: 115153

LOAD_AUGMENTED_DATA = True
PARALLEL = False
##############################################################################


def createNumpyData(file_list, file_to_save):
    """Create a numpy array for the iiw data."""
    l = len(file_list)

    images_list = []

    # images_blob = np.empty((l, 3, RESIZE_HEIGHT, RESIZE_WIDTH))
    # # comparisons are saved the following way:
    # # file_num x regular / augmented data x comparison_index x one_comparison
    comparisons_blob = np.full((l, MAX_NUM_COMPARISONS+1, 1, 6), np.nan)
    # blob_to_get_size = np.full((l, 1, 1, 6), np.nan)
    #
    # gb = 2**30
    # img_blob_size = images_blob.nbytes / gb
    # comp_blob_size = comparisons_blob.nbytes / gb
    # aug_blob_size = blob_to_get_size.nbytes * (MAX_NUM_AUGMENTED+1) / gb
    # tot_lists_size = 8 * l * 4 / gb
    #
    # print("Estimated total amount of memory needed to run:",
    #       (img_blob_size + comp_blob_size + aug_blob_size + tot_lists_size),
    #       "GB.")

    if AUGMENT_DATA:
        augmented_blob = np.full((l, MAX_NUM_AUGMENTED+1, 1, 6), np.nan)
    else:
        augmented_blob = np.zeros((l, 1, 1, 6))

    # print("Sizes of blobs:",
    #       "images:", img_blob_size, "GB,",
    #       "comparisons:", comp_blob_size, "GB,",
    #       "augmented comparisons:", aug_blob_size, "GB.")

    # get statistics of number of comparisons in all images
    num_comparisons = []
    num_augmented_comparisons = []
    heights = []  # save dimensions to get statistics
    widths = []

    print("\n\nTotal number of files:", l)

    if PARALLEL:
        pbar = ProgressBar(maxval=l,
                           widgets=[SimpleProgress(), ': ',
                                    Percentage(), ' ',
                                    Bar('=', '[', ']'), ' ',
                                    ETA()])
        pbar.start()

        def mycallback(result):
            img, cb, ab, h, w, nc, na, fc = result
            num_comparisons.append(nc)
            num_augmented_comparisons.append(na)
            heights.append(h)
            widths.append(w)
            # images_blob[fc, :, :, :] = img
            images_list.append(img)
            comparisons_blob[fc, :, 0, :] = cb
            try:
                augmented_blob[fc, :, 0, :] = ab
            except IndexError:
                print("Need at least MAX_NUM_AUGMENTED =", na)

        num_processes = multiprocessing.cpu_count()  # - 1
        print("using", num_processes, "cores to go through the file list")
        pool = multiprocessing.Pool(num_processes)
        _res = [pool.apply_async(getDataForSingleFile, (file_count, file_name),#noqa
                                 callback=mycallback)  # sub_results.append)
                for file_count, file_name in enumerate(file_list)]

        while len(num_comparisons) != l:
            pbar.update(len(num_comparisons))
            time.sleep(1.0)
        pbar.finish()
    else:
        # single thread: loop through every image in the folder
        for fc_i, fn in enumerate(tqdm(file_list)):
            img, cb, ab, h, w, nc, na, fc_o = getDataForSingleFile(fc_i, fn)
            num_comparisons.append(nc)
            num_augmented_comparisons.append(na)
            heights.append(h)
            widths.append(w)
            images_list.append(img)
            # images_blob[fc_i, :, :, :] = img
            comparisons_blob[fc_i, :, 0, :] = cb
            try:
                augmented_blob[fc_i, :, 0, :] = ab
            except IndexError:
                print("Need at least MAX_NUM_AUGMENTED =", na)
            # pbar.update(fc_i)
        # pbar.finish()

    # show some statistics
    portrait = 0
    landscape = 0
    square = 0
    aspect_ratios = []
    for i in range(len(heights)):
        h = heights[i]
        w = widths[i]
        aspect_ratios.append(w/h)
        if h > w:
            portrait += 1
        elif h < w:
            landscape += 1
        else:
            square += 1
    # print("aspect ratios:", sorted(aspect_ratios))
    # print("comparisons:", sorted(num_comparisons))
    # print("augmented:", sorted(num_augmented_comparisons))
    # print("heights:", sorted(heights))
    # print("widths:", sorted(widths))

    # all images have between 1 and 1181 comparisons
    # and between 1 and 60048 augmented
    print("\ncomparisons:\nmin:", min(num_comparisons),
          "\tmax:", max(num_comparisons),
          "\tavg:", np.mean(num_comparisons),
          "\tmedian:", np.median(num_comparisons),
          "\naugmented comparisons:\nmin:", min(num_augmented_comparisons),
          "\tmax:", max(num_augmented_comparisons),
          "\tavg:", np.mean(num_augmented_comparisons),
          "\tmedian:", np.median(num_augmented_comparisons),
          "\nheight:\nmin:", min(heights),
          "\tmax:", max(heights),
          "\tavg:", np.mean(heights),
          "\tmedian:", np.median(heights),
          "\nwidth:\nmin:", min(widths),
          "\tmax:", max(widths),
          "\tavg:", np.mean(widths),
          "\tmedian:", np.median(widths),
          "\naspect ratio:\nmin:", min(aspect_ratios),
          "\tmax:", max(aspect_ratios),
          "\tavg:", np.mean(aspect_ratios),
          "\tmedian:", np.median(aspect_ratios))
    print("portrait:", portrait, "landscape:", landscape, "square:", square)
    total_num_comparisons = sum(num_comparisons)
    total_num_augmented = sum(num_augmented_comparisons)
    print("Total number of comparisons:", total_num_comparisons)
    print("Total number of augmented comparisons:", total_num_augmented)
    print("So we improved the comparisons by a factor of",
          total_num_augmented / total_num_comparisons)


    print("please be patient, images are resized!")

    for height, width in RESOLUTIONS:
        images_blob = convert_images_to_blob(images_list, height, width)
        # slower in saving but faster in loading:
        full_path = file_to_save + "_{}_{}_sRGB".format(height, width)
        print("Saving file:", full_path)
        save_start = timeit.default_timer()
        np.savez_compressed(full_path,
                            images=threshold(images_blob),
                            comparisons=comparisons_blob,
                            augmented=augmented_blob,
                            )
        save_end = timeit.default_timer()
        print('Saving the file took', save_end-save_start, 'seconds')
    for height, width in RESOLUTIONS:
        images_blob = convert_images_to_blob(images_list, height, width)
        # slower in saving but faster in loading:
        full_path = file_to_save + "_{}_{}_linear".format(height, width)
        print("Saving file:", full_path)
        save_start = timeit.default_timer()
        np.savez_compressed(full_path,
                            images=threshold(srgb_to_rgb(images_blob)),
                            comparisons=comparisons_blob,
                            augmented=augmented_blob,
                            )
        save_end = timeit.default_timer()
        print('Saving the file took', save_end-save_start, 'seconds')

    end = timeit.default_timer()
    print('The creation took', end-start, 'seconds')


def convert_images_to_blob(images_list, height, width):
    """Resize image and convert list to numpy array."""
    images_blob = np.empty((len(images_list), 3, height, width))
    # pbar = ProgressBar(maxval=len(images_list),
    #                    widgets=[SimpleProgress(), ': ',
    #                             Percentage(), ' ',
    #                             Bar('=', '[', ']'), ' ',
    #                             ETA()])
    # pbar.start()

    for i, image in enumerate(tqdm(images_list)):
        # resize the image
        # pay attention to ordering of tuple in cv2 vs scipy
        # image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT),
        #                    interpolation=cv2.INTER_AREA)
        # scipy gives RGB image (not BGR as in cv2)
        resized = scipy.misc.imresize(image, (height, width))
        images_blob[i, :, :, :] = np.transpose(resized / 255, (2, 0, 1))
        # now image is a 0-1 float linear RGB image with
        # channels x height x width
        # images_blob[file_count, :, :, :] = image
        # pbar.update(i)
    # pbar.finish()
    return images_blob


def threshold(images_blob):
    """Threshold images to avoid 0."""
    # np.finfo(np.float32).eps = 1.1920929e-07
    # np.finfo(np.float).eps = 2.2204460492503131e-16
    return np.maximum(images_blob, 1e-5)


def getDataForSingleFile(file_count, file_name):
    """Create the data (image, comparisons, augmented) for one image."""
    # print("working on file ", file_count,
    #       "file name: ", file_name)

    # load image
    # image = cv2.imread(DATA_FOLDER + file_name + IMAGE_EXTENSION)
    # use scipy to get RGB instead of cv2 BGR with conversion to RGB
    image = scipy.misc.imread(os.path.join(DATA_FOLDER,
                                           file_name + IMAGE_EXTENSION))

    # get size for statistics
    # cv2 saves image as height, width, channels = image.shape
    height = image.shape[0]
    width = image.shape[1]

    # load json description according to image
    json_data = os.path.join(DATA_FOLDER, file_name + '.json')
    # prepare a list for the comparisons
    comparisons = []
    points = {}
    with open(json_data) as json_data:
        data = json.load(json_data)

        # for statistics
        num_comparisons = len(data["intrinsic_comparisons"])
        # print("number of comparisons:", num_comparisons)

        # first save a dictionary of the point ids
        # used in that image, needed to find according points
        # that are compared in the list of intrinsic points

        # the dictionary should look like:
        # points = {id1: [x1, y1, opaque1], id2: [x2, y2, opaque2], ...}
        for point in data["intrinsic_points"]:
            points[point["id"]] = [point["x"], point["y"], point["opaque"]]

        # go through all intrinsic comparisons
        for comparison in data["intrinsic_comparisons"]:
            point1 = comparison["point1"]
            point2 = comparison["point2"]
            darker = comparison["darker"]
            darker_score = comparison["darker_score"]
            # change darker judgment to number
            switch = {"1": 1, "2": 2, "E": 0}
            darker_num = switch[darker]
            comparisons.append([point1, point2, darker_num, darker_score])
        # done reading comparisons
    # done reading from json

    # print(comparisons)
    comparisons_blob = comparisons_to_matrix(comparisons,
                                             file_name,
                                             points,
                                             MAX_NUM_COMPARISONS)

    if AUGMENT_DATA:
        augmented_file = os.path.join(DATA_FOLDER,
                                      file_name + '_augmented' + '.json')
        comparisons_file = os.path.join(DATA_FOLDER,
                                        file_name + '_comparisons' + '.json')
        # if we should not regenerate all augmented data and find the file
        # from a previous run, load it
        if LOAD_AUGMENTED_DATA and os.path.isfile(augmented_file):
            # print("loading file", augmented_file)
            with open(augmented_file, 'r') as infile:
                augmented = json.load(infile)
                # check if loading gives the same result
                # Note: should be commented, otherwise it does not make sense
                # to save and loater load, the saving is just to not compute
                # it over and over again!
                # assert(augmented == augment(comparisons))
                # print("COMMENT ABOVE LINE")
        else:
            # print("creating:", augmented_file)
            augmented = augment(comparisons)
            # print("saving file", augmented_file)
            # save data to json
            with open(augmented_file, 'w') as outfile:
                json.dump(augmented, outfile)
            # save comparisons for comparison with augmented data
            # in a json file in "my" format:
            with open(comparisons_file, 'w') as outfile:
                json.dump(comparisons, outfile)
        num_augmented_comparisons = len(augmented)
        # print("number of augmented comparisons are",
        #       num_augmented_comparisons,
        #       "instead of",
        #       num_comparisons,
        #       "so we gained an improvement of",
        #       num_augmented_comparisons / num_comparisons)

        augmented_blob = comparisons_to_matrix(augmented,
                                               file_name,
                                               points,
                                               MAX_NUM_AUGMENTED)
    else:
        augmented_blob = np.zeros((1, 6))
        num_augmented_comparisons = 0

    return (image,
            comparisons_blob,
            augmented_blob,
            height,
            width,
            num_comparisons,
            num_augmented_comparisons,
            file_count,
            )


def unify(comparisons, weights='actual', threshold=0.5):
    """Unify all comparisons in one way and with appropriate weights.

    Unify all comparisons:
    a = b => a = b and b = a
    a < b => a < b
    a > b => b < a
    where 0 means = and 1 means <.
    The appropriate weight can be chose either as
    - actual: the given weight of the human judgement
    - thresholded: 1 if it is above a threshold and 0 otherwise
                   (removed from comparisons).
    """
    unified = []

    if weights == 'actual':
        for c in comparisons:
            if c[2] == 0:
                unified.append((c[0], c[1], 0, c[3]))
                unified.append((c[1], c[0], 0, c[3]))
            elif c[2] == 1:
                unified.append((c[1], c[0], 2, c[3]))
            elif c[2] == 2:
                unified.append((c[0], c[1], 2, c[3]))
            else:
                raise Exception('Expecting 0,1,2 as comparison, got', c[2])
    elif weights == 'thresholded':
        print("Using threshold", threshold)
        for c in comparisons:
            if c[3] > threshold:
                if c[2] == 0:
                    unified.append((c[0], c[1], 0, 1))
                    unified.append((c[1], c[0], 0, 1))
                elif c[2] == 1:
                    unified.append((c[1], c[0], 2, 1))
                elif c[2] == 2:
                    unified.append((c[0], c[1], 2, 1))
                else:
                    raise Exception('Expecting 0,1,2 as comparison, got', c[2])
    else:
        raise Exception("Method", weights, "to apply for the weights "
                        "is not known.")

    # print("before and after changing the comparisons to one way\n",
    #       comparisons,
    #       unified)
    return unified


def augment(comparisons, weights='actual', consolidationMethod='min'):
    """Augment the comparisons by adding the transitive hull."""
    unified = unify(comparisons, weights)

    # build dictionary of pointIDs <-> nodeIDs and create matrix
    point_to_node = {}
    node_to_point = []
    for x, y, r, w in unified:
        if x not in point_to_node:
            point_to_node[x] = len(node_to_point)
            node_to_point.append(x)
        if y not in point_to_node:
            point_to_node[y] = len(node_to_point)
            node_to_point.append(y)

    # # one quick check:
    # index = int(len(node_to_point) * np.random.rand())
    # assert(index == point_to_node[node_to_point[index]])

    # create adjacency matrix:
    n = len(node_to_point)
    matrix = np.full((2, n, n), np.nan)
    for x, y, r, w in unified:
        matrix[0, point_to_node[x], point_to_node[y]] = r
        matrix[1, point_to_node[x], point_to_node[y]] = w

    # np.set_printoptions(suppress=True, precision=3, threshold=1000)
    # print("before warshall\n", matrix)
    # time_last = timeit.default_timer()

    # compute transitive closure by Floyd-Warshall algorithm
    time_last = timeit.default_timer()
    matrix = warshall(matrix, consolidationMethod)
    print("Time taken:", timeit.default_timer() - time_last)

    # print("after warhsall\n", matrix, timeit.default_timer() - time_last)

    # go back to pointIDs and a list of comparisons
    augmented = []
    for i in range(n):
        for j in range(n):
            if np.isfinite(matrix[0, i, j]):
                augmented.append([node_to_point[i],
                                  node_to_point[j],
                                  matrix[0, i, j],
                                  matrix[1, i, j]])

    return augmented


def consolidate(wik, wkj, method='min'):
    """Consolidate the weights wik and wkj by appropriate mixing.

    Mix the weights by one of the following methods:
    - arithmetic mean
    - geometric mean
    - minimum
    The important property is that the result must be nan if one of the inputs
    is nan.
    """
    # print("Consolidation Method:", method)
    if method == 'min':
        # minimum of the values
        if np.isnan(wik) or np.isnan(wkj):
            return np.nan
        else:
            return min(wik, wkj)
    elif method == 'arithmeticMean':
        return (wik + wkj) / 2  # arithmetic mean
    elif method == 'geometricMean':
        return (wik * wkj)**0.5  # gemetric mean
    else:
        raise Exception("Method", method, "is not known.")


def warshall(a, consolidationMethod='min'):
    """Use Floyd-Warshall algorithm to find the transitive hull."""
    # print("Consolidation Method:", consolidationMethod)
    n = a.shape[1]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != j:
                    wik = a[1, i, k]
                    wkj = a[1, k, j]
                    wij = a[1, i, j]
                    wij_new = consolidate(wik, wkj, consolidationMethod)
                    # if there is a connection from i to j via k then the above
                    # sum results in a finite number. So if additionaly there
                    # was no connection from i to j yet or there was but the
                    # new weight is higher, then set the new connection
                    if np.isfinite(wij_new) and \
                            (np.isnan(wij) or wij < wij_new):
                        # get the correct relation ('<' or '=')
                        if a[0, i, k] == a[0, k, j]:
                            # both '=' or both '<', then use '=' or '<'
                            a[0, i, j] = a[0, i, k]
                        else:  # one '=', one '<', then use '<'
                            a[0, i, j] = 2
                        # set the weight
                        a[1, i, j] = wij_new
                        # print((i, k, a[0, i, k], a[1, i, k]),
                        #       (k, j, a[0, k, j], a[1, k, j]),
                        #       (i, j, a[0, i, j], a[0, i, j]),
                        #       (consolidate(wik, wkj, 'min'),
                        #        consolidate(wik, wkj, 'geometricMean'),
                        #        consolidate(wik, wkj, 'arithmeticMean')))

    # do a consistency check (tuples with A<B and B<A or A<B and B=A)
    failed_consistency = 0
    biggest = 0
    for i in range(n):
        for j in range(n):
            # if there is a comparison for i and j, but they are not consistent
            if (a[0, i, j] == 2 and a[0, j, i] == 2) or \
                    (a[0, i, j] == 2 and a[0, j, i] == 0) or\
                    (a[0, i, j] == 0 and a[0, j, i] == 2):
                failed_consistency += 1
                # print("failed consistency check:", i, j,
                #       "certainty:", a[1, i, j], a[1, j, i])

                # only keep the comparison with the higher weight
                if a[1, i, j] > a[1, j, i]:
                    if biggest < a[1, j, i]:
                        biggest = a[1, j, i]
                    a[:, j, i] = np.nan
                else:
                    if biggest < a[1, i, j]:
                        biggest = a[1, i, j]
                    a[:, i, j] = np.nan

            # if we have 'A=B' and 'B=A', we remove one
            if a[0, i, j] == 0 and a[0, j, i] == 0:
                # only keep one '=' comparison,
                # this is no failed consistency, just to remove bias towards
                # '=' comparisons
                # pick by chance
                rand = np.random.rand()
                if rand > 0.5:
                    a[:, j, i] = np.nan
                else:
                    a[:, i, j] = np.nan
    print("Removed", failed_consistency,
          "comparisons with highest certainty {:4.2f}".format(biggest),
          "(failed consistency check). Left:",
          np.sum(np.isfinite(a[0, :, :])),
          "with certainty range",
          "[{:4.2f}, {:4.2f}]".format(np.nanmin(a[1, :, :]),
                                      np.nanmax(a[1, :, :])))

    # NOTE: the input a itself is changed, so in general it would not need
    # to be returned!
    return a


def comparisons_to_matrix(comparisons, file_name, points, max_size):
    """Create matrix from comparisons.

    Create matrix from comparisons, where the c-th comparison is
    stored in comparisons_blob[c. :]
    and last c is reserved for meta data (num_comparisons, file_name).
    """
    comparisons_blob = np.full((max_size+1, 6), np.nan)

    # add comparisons
    for c, (point1, point2, darker, weight) in enumerate(comparisons):
        # get x and y from the dictionary
        x1, y1, opaque1 = points[point1]
        x2, y2, opaque2 = points[point2]

        # # don't know yet, what to do with the opaque information.
        # # Leave the not opaque ones out?
        # if not (opaque1 and opaque2):
        #     print("not opaque")  # seems like this never happens

        # save in numpy array
        comparisons_blob[c, 0] = x1
        comparisons_blob[c, 1] = y1
        comparisons_blob[c, 2] = x2
        comparisons_blob[c, 3] = y2
        comparisons_blob[c, 4] = darker
        comparisons_blob[c, 5] = weight

    # add meta data
    comparisons_blob[max_size, 0] = len(comparisons)
    comparisons_blob[max_size, 1] = float(file_name)
    comparisons_blob[max_size, 2] = 0  # coding for read comparisons from blob

    return comparisons_blob


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


def randomlySplitIntoTwoLists(file_names):  # using SPLIT
    """Randomly split the content of the input list into two lists.

    Create two lists and put SPLIT amout of the items of file_names into first
    and the rest into second list by using random generator.
    """
    train = []
    test = []
    for fn in file_names:
        rand = np.random.rand(1)
        if rand < SPLIT:
            train.append(fn)
        else:
            test.append(fn)
    return train, test


def narihiraSplitIntoTwoLists(file_names):  # using Narihira's method
    """Split the content as in the Narihira paper and put into two lists."""
    train = []
    test = []
    for ind, fn in enumerate(file_names):
        if ind % 5:
            train.append(fn)
        else:
            test.append(fn)
    return train, test


def narihiraSplitIntoThreeLists(file_names):  # using Narihira's method
    """Split the content as in the Narihira paper and put into three lists."""
    train = []
    val = []
    test = []
    for ind, fn in enumerate(file_names):
        if ind % 5 == 0:  # take ID 0 and 5 from 0-9 (20%)
            test.append(fn)
        elif ind % 10 == 6:  # take ID 6 from 0-9 (10%)
            val.append(fn)
        else:
            train.append(fn)  # take ID 1,2,3,4,7,8,9 (70%)
    return train, val, test


def splitIntoBigTrainMiniVal(file_names):  # using Narihira's method
    """Split the content as in the Narihira paper and put into three lists."""
    train = []
    val = []
    test = []
    for ind, fn in enumerate(file_names):
        if ind % 5 == 0:  # take ID 0 and 5 from 0-9 (20%)
            test.append(fn)
        elif ind % 100 == 6:  # take ID 6 from 0-99 (1%)
            val.append(fn)
        else:
            train.append(fn)  # take ID 1,2,3,4,7,8,9, 11-14, 16-19, ... (79%)
    return train, val, test


if __name__ == "__main__":
    # test the consolidate function:
    assert(np.isnan(consolidate(np.nan, np.random.rand())))
    assert(np.isnan(consolidate(np.random.rand(), np.nan)))
    assert(np.isnan(consolidate(np.nan, np.nan)))
    assert(np.isfinite(consolidate(np.random.rand(), np.random.rand())))

    # get list of all names of images in folder
    file_names = [os.path.splitext(file_name)[0]  # without extension
                  for file_name in os.listdir(DATA_FOLDER)
                  if file_name.endswith(IMAGE_EXTENSION)]
    # the following is important on Ubuntu,
    # otherwise os.listdir gives unordered list
    # but for a consistent test set we need them sorted!
    file_names.sort()
    print("All files in folder:", file_names)

    for create_now in CREATE:

        file_to_save = SAVE_TO_FOLDER + '/' + create_now

        if create_now == 'dummy':
            # use 20 random images, 10 for train, 10 for test
            # file_list = []
            # for i in range(20):
            #     # print("TAKING RANDOM IMAGES FOR DUMMY!!!")
            #     rand = np.random.randint(len(file_names))
            #     file_list.append(file_names[rand])

            # use the first 20 images, 10 for train, 10 for test
            file_list = file_names[:20]
            print("File list for dummy val:", file_list[:10])
            file_to_save = SAVE_TO_FOLDER + 'dummy_val'
            createNumpyData(file_list[:10], file_to_save)
            print("File list for dummy train:", file_list[10:])
            file_to_save = SAVE_TO_FOLDER + 'dummy_train'
            createNumpyData(file_list[10:], file_to_save)
        elif create_now == 'one':  # ['118495']
            file_to_save = SAVE_TO_FOLDER + 'one_train'
            createNumpyData(['110701'], file_to_save)
            file_to_save = SAVE_TO_FOLDER + 'one_test'
            createNumpyData(['24929'], file_to_save)

            file_to_save = SAVE_TO_FOLDER + 'two_train'
            createNumpyData(['110701', '24929'], file_to_save)
            file_to_save = SAVE_TO_FOLDER + 'two_test'
            createNumpyData(['110701', '24929'], file_to_save)

        elif create_now == 'all':
            print("A lot of memory needs to be",
                  "allocated, please be patient.")
            file_list = file_names
            createNumpyData(file_list, file_to_save)
        elif create_now == 'allShuffled':
            print("A lot of memory needs to be",
                  "allocated, please be patient.")
            file_list = file_names
            np.random.shuffle(file_list)
            createNumpyData(file_list, file_to_save)
        elif create_now == 'trainTest':
            if NARIHIRA:
                print("Split using Narihira's method")
                train, test = narihiraSplitIntoTwoLists(file_names)
            else:
                print("Split using random drawings")
                train, test = randomlySplitIntoTwoLists(file_names)
            print("Train files:", train, "\nTest files:", test)
            print("A lot of memory needs to be",
                  "allocated, please be patient.")
            file_to_save = SAVE_TO_FOLDER + 'train'
            createNumpyData(train, file_to_save)
            file_to_save = SAVE_TO_FOLDER + 'test'
            createNumpyData(test, file_to_save)
        elif create_now == 'trainValTest':
            if NARIHIRA:
                print("Split similar to Narihira's method")
                train, val, test = narihiraSplitIntoThreeLists(file_names)
            else:
                print("Provide how to split the data")
            # print("Train files:", train, "\nTest files:", test)
            print("Train files:", len(train),
                  "\nValidation files:", len(val),
                  "\nTest files:", len(test))
            print("A lot of memory needs to be",
                  "allocated, please be patient.")
            file_to_save = SAVE_TO_FOLDER + 'trainValTest_train'
            createNumpyData(train, file_to_save)
            file_to_save = SAVE_TO_FOLDER + 'trainValTest_val'
            createNumpyData(val, file_to_save)
            file_to_save = SAVE_TO_FOLDER + 'trainValTest_test'
            createNumpyData(test, file_to_save)
        elif create_now == 'bigTrainMiniValTest':
            if NARIHIRA:
                print("Split similar to Narihira's method")
                train, val, test = splitIntoBigTrainMiniVal(file_names)
            else:
                print("Provide how to split the data")
            print("Train files:", len(train),
                  "\nValidation files:", len(val),
                  "\nTest files:", len(test))
            print("A lot of memory needs to be",
                  "allocated, please be patient.")
            file_to_save = SAVE_TO_FOLDER + 'bigTrainMiniValTest_train'
            createNumpyData(train, file_to_save)
            file_to_save = SAVE_TO_FOLDER + 'bigTrainMiniValTest_val'
            createNumpyData(val, file_to_save)
            file_to_save = SAVE_TO_FOLDER + 'bigTrainMiniValTest_test'
            createNumpyData(test, file_to_save)
        else:
            raise NameError('DATA was ' + create_now +
                            'but should be one of the following: ' +
                            'dummy, one, trainTest, trainValTest,  all')
