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

"""Defines several tools for monitoring net activity."""
# from __future__ import division
# Checkpointer below depends on regular division!!!
from __future__ import print_function

# import pdb

import logging
import shutil
import os as _os
import numpy as np
from barrista.monitoring import Monitor

_LOGGER = logging.getLogger(__name__)


# mostly taken from the Checkpointer of barrista
class CheckpointerIncludingRename(Monitor):  # pylint: disable=R0903
    r"""
    Writes the network blobs to disk at certain iteration intervals.

    The logger makes use of the following keyword arguments
    (\* indicates required):

    * ``iter``\*,
    * ``net``\*,
    * ``batch_size``\*.

    :param name_prefix: string or None.
      The first part of the output filenames to generate. The prefix '_iter_,
      the current iteration, as well as '.caffemodel' is added.

      If you are using a caffe version from later than Dec. 2015, caffe's
      internal snapshot method is exposed to Python and also snapshots the
      solver. If it's available, then this method will be used. However,
      in that case, it's not possible to influence the storage location
      from Python. Please use the solver parameter ``snapshot_prefix``
      when constructing the solver instead (this parameter may be None
      and is unused then).

    :param iterations: int > 0.
      Always if the current number of iterations is divisible by iterations,
      the network blobs are written to disk. Hence, this value must be a
      multiple of the batch size!
    """

    def __init__(self,
                 name_prefix,
                 iterations,
                 base_iterations=0):
        """See class documentation."""
        assert iterations > 0
        _LOGGER.info('Setting up checkpointing with name prefix %s every ' +
                     '%d iterations.', name_prefix, iterations)
        self.name_prefix = name_prefix
        self.iterations = iterations
        self.created_checkpoints = []
        self._base_iterations = base_iterations

    # pylint: disable=arguments-differ
    def _post_train_batch(self, kwargs, finalize=False):
        assert self.iterations % kwargs['batch_size'] == 0, (
            'iterations not multiple of batch_size, {} vs {}'.format(
                self.iterations, kwargs['batch_size']))
        # Prevent double-saving.
        if kwargs['iter'] in self.created_checkpoints:
            return
        if ((kwargs['iter'] + self._base_iterations +
             kwargs['batch_size']) % self.iterations == 0 or
                finalize):
            self.created_checkpoints.append(kwargs['iter'])
            # pylint: disable=protected-access
            if not hasattr(kwargs['solver']._solver, 'snapshot'):  # pragma: no cover
                checkpoint_filename = (
                    self.name_prefix + '_iter_' +
                    str(int((kwargs['iter'] + self._base_iterations) /
                            kwargs['batch_size']) + 1) +
                    '.caffemodel')
                _LOGGER.debug("Writing checkpoint to file '%s'.",
                              checkpoint_filename)
                kwargs['net'].save(checkpoint_filename)
            else:
                # pylint: disable=protected-access
                kwargs['solver']._solver.snapshot()
                caffe_checkpoint_filename = (self.name_prefix +
                                             '_iter_' +
                                             str((kwargs['iter'] + self._base_iterations) /
                                                 kwargs['batch_size'] + 1) +
                                             '.caffemodel')
                caffe_sstate_filename = (self.name_prefix +
                                         '_iter_' +
                                         str((kwargs['iter'] + self._base_iterations) /
                                             kwargs['batch_size'] + 1) +
                                         '.solverstate')
                _LOGGER.debug('Writing checkpoint to file "[solverprefix]%s" ' +
                              'and "[solverprefix]%s".',
                              caffe_checkpoint_filename,
                              caffe_sstate_filename)
                assert _os.path.exists(caffe_checkpoint_filename), (
                    "An error occured checkpointing to {}. File not found. "
                    "Make sure the `base_iterations` and the `name_prefix` "
                    "are correct.").format(caffe_checkpoint_filename)
                assert _os.path.exists(caffe_sstate_filename), (
                    "An error occured checkpointing to {}. File not found. "
                    "Make sure the `base_iterations` and the `name_prefix` "
                    "are correct.").format(caffe_sstate_filename)

                # tnestmeyer addition:
                # since the checkpoint is after the batch processing,
                # add the number of processed samples (batch_size)
                # to the counter
                barrista_checkpoint_filename = (self.name_prefix +
                                                '_barrista_iter_' +
                                                str(kwargs['iter'] +
                                                    kwargs['batch_size']) +
                                                '.caffemodel')
                barrista_sstate_filename = (self.name_prefix +
                                            '_barrista_iter_' +
                                            str(kwargs['iter'] +
                                                kwargs['batch_size']) +
                                            '.solverstate')
                # now rename the caffemodels and solverstates
                _LOGGER.debug('Renaming checkpoint to file ' +
                              '"%s" and solver to "%s".',
                              barrista_checkpoint_filename,
                              barrista_sstate_filename)
                shutil.move(caffe_checkpoint_filename,
                            barrista_checkpoint_filename)
                shutil.move(caffe_sstate_filename,
                            barrista_sstate_filename)

    def finalize(self, kwargs):
        """Write a final checkpoint."""
        # Account for the counting on iteration increase for the last batch.
        kwargs['iter'] -= kwargs['batch_size']
        self._post_train_batch(kwargs, finalize=True)
        kwargs['iter'] += kwargs['batch_size']

        # tnestmeyer addition:
        # remember that iter always had the value before processing the batch
        _LOGGER.debug('All checkpointed iterations: %s',
                      [checkpoint + kwargs['batch_size']
                       for checkpoint in self.created_checkpoints])


class CombineLosses(Monitor):
    """
    Combine the losses needed for the WHDR CNN.

    Compute the weighted sum of the given losses.
    """

    def __init__(self,
                 loss_scale_whdr,
                 loss_scale_lambert):
        """Initialize the weights."""
        self.scale_whdr = loss_scale_whdr
        self.scale_lambert = loss_scale_lambert

    def _post_test(self, kwargs):
        self._post_combinedLoss(kwargs)

    def _post_train_batch(self, kwargs):
        self._post_combinedLoss(kwargs)

    def _post_combinedLoss(self, kwargs):
        loss = 0.0
        if self.scale_whdr:
            loss += self.scale_whdr * kwargs['loss_whdr_hinge']
        if self.scale_lambert:
            loss += self.scale_lambert * kwargs['loss_lambert']
        kwargs['loss_combined'] = loss


class RunningAverage(Monitor):
    """
    Show smoothed WHDR.
    """

    def __init__(self,
                 train_size,
                 batch_size):
        """Initialize the weights."""
        self.num_samples = int(train_size / batch_size)
        self.cycle_whdrs = np.full(self.num_samples, np.nan)
        print("Show running average of WHDR over",
              self.num_samples,
              "samples")

    def _post_train_batch(self, kwargs):
        # iter will always be a multiple of batch_size after processing a batch
        index = int(kwargs['iter']/kwargs['batch_size']) % self.num_samples
        self.cycle_whdrs[index] = kwargs['whdr_original']
        # post to kwargs
        # kwargs['running_average'] = np.mean(self.whdrs)
        kwargs['running_average'] = np.nanmean(self.cycle_whdrs)


class WHDRProgressIndicator(Monitor):
    r"""
    Progress indicator showing info for WHDR project.

    Implement a ProgressIndicator similar to the one in barrista, providing all
    the details needed to check the progress during the learning phase to
    later predict reflectance images with low WHDR. Mainly builds on the
    _WHDRIndicator.
    """

    def __init__(self,
                 loss_scale_whdr,
                 loss_scale_boundaries01,
                 loss_scale_lambert):
        """See class documentation."""
        self.loss_combined = None
        self.loss_whdr_hinge = None
        self.loss_whdr_hinge_level0 = None
        self.loss_boundaries_refl = None
        self.loss_boundaries_shad = None
        self.loss_lambert = None
        self.whdr_original = None
        self.whdr_original_level0 = None
        self.test_whdr = None
        self.running_average = np.nan
        from progressbar import ETA, Percentage, SimpleProgress, ProgressBar
        self.widgets = ['|Iter: ', SimpleProgress(), ' =', Percentage(),
                        ' ', ETA()]
        self.pbarclass = ProgressBar
        self.pbar = None
        # Initialize the weights.
        self.scale_whdr = loss_scale_whdr
        self.scale_boundaries01 = loss_scale_boundaries01
        self.scale_lambert = loss_scale_lambert

    def _post_train_batch(self, kwargs):
        if self.pbar is None:
            # see below for _WHDRIndicator
            widgets = [_WHDRIndicator(self)] + self.widgets
            self.pbar = self.pbarclass(maxval=kwargs['max_iter'],
                                       widgets=widgets)
            self.pbar.start()
        keys = list(kwargs.keys())
        if 'whdr_original' in keys:
            self.whdr_original = kwargs['whdr_original']
        if 'loss_whdr_hinge' in keys:
            self.loss_whdr_hinge = self.scale_whdr * kwargs['loss_whdr_hinge']
        if 'whdr_original_level0' in keys:
            self.whdr_original_level0 = kwargs['whdr_original_level0']
        if 'loss_whdr_hinge_level0' in keys:
            self.loss_whdr_hinge_level0 = (self.scale_whdr *
                                           kwargs['loss_whdr_hinge_level0'])
        if 'loss_boundaries_reflectance' in keys:
            self.loss_boundaries_refl = (self.scale_boundaries01 *
                                         kwargs['loss_boundaries_reflectance'])
        if 'loss_boundaries_reflectance' in keys:
            self.loss_boundaries_shad = (self.scale_boundaries01 *
                                         kwargs['loss_boundaries_shading'])
        if 'loss_lambert' in keys:
            self.loss_lambert = (self.scale_lambert * kwargs['loss_lambert'])
        if 'running_average' in keys:
            self.running_average = kwargs['running_average']
        if 'loss_combined' in keys:
            self.loss_combined = kwargs['loss_combined']
        self.pbar.update(value=kwargs['iter'])

    def _post_test(self, kwargs):
        if 'whdr_original' in list(kwargs.keys()):
            self.test_whdr = kwargs['whdr_original']

    def finalize(self, kwargs):
        """Call ``progressbar.finish()``."""
        if self.pbar is not None:
            self.pbar.finish()


# inspired by the ProgressIndicator of barrista
class _WHDRIndicator(object):
    r"""
    A plugin indicator for the ``progressbar`` package.

    This must be used in conjunction with the
    :py:class:`barrista.monitoring.ProgressIndicator`. If available, it
    outputs current loss, accuracy, test loss and test accuracy.

    :param progress_indicator:
      :py:class:`barrista.monitoring.ProgressIndicator`. The information
      source to use.
    """

    def __init__(self, progress_indicator):
        self.progress_indicator = progress_indicator
        print("Loss=loss_combined,",
              "Hinge=loss_whdr_hinge,",
              "BR=boundaries01 loss on reflectance,",
              "BS=boundaries01 loss on shading,",
              "Lamb=loss_lambert,",
              "WHDR=whdr_original,",
              "RunAvg=running average over epoch,",
              "Test=Test WHDR.")

    def __call__(self, pbar, stats):
        r"""Compatibility with new versions of ``progressbar2``."""
        return self.update(pbar)

    def update(self, pbar):
        """The update method to implement by the ``progressbar`` interface."""
        ret_val = ''
        # print(pbar)
        if self.progress_indicator.loss_combined is not None:
            ret_val += '\n|Loss: {0:.4f}'.format(
                self.progress_indicator.loss_combined)
        if self.progress_indicator.loss_whdr_hinge_level0 is not None:
            ret_val += '|HL0: {0:.2f}'.format(
                self.progress_indicator.loss_whdr_hinge_level0)
        if self.progress_indicator.loss_whdr_hinge is not None:
            ret_val += '|Hinge: {0:.2f}'.format(
                self.progress_indicator.loss_whdr_hinge)
        if self.progress_indicator.loss_boundaries_refl is not None:
            ret_val += '|BR: {0:.2f}'.format(
                self.progress_indicator.loss_boundaries_refl)
        if self.progress_indicator.loss_boundaries_shad is not None:
            ret_val += '|BS: {0:.2f}'.format(
                self.progress_indicator.loss_boundaries_shad)
        if self.progress_indicator.loss_lambert is not None:
            ret_val += '|Lamb: {0:.4f}'.format(
                self.progress_indicator.loss_lambert)
        if self.progress_indicator.whdr_original_level0 is not None:
            ret_val += '|WL0: {0:5.2f}'.format(
                self.progress_indicator.whdr_original_level0 * 100)
        if self.progress_indicator.whdr_original is not None:
            ret_val += '|WHDR: {0:5.2f}'.format(
                self.progress_indicator.whdr_original * 100)
        if not np.isnan(self.progress_indicator.running_average):
            ret_val += '|RunAvg: {0:5.2f}'.format(
                self.progress_indicator.running_average * 100)
        if self.progress_indicator.test_whdr is not None:
            ret_val += '|Test: {0:.4f}'.format(
                self.progress_indicator.test_whdr)
        return ret_val
