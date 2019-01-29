"""
Module to compute the
+ Inverse Consistency (IC)
+ Scalar Associativity (SA)
+ Stepwise Error (SE)
for the selected data set, among the ones generated in the previous experiments bm1_ to bm6_ .
These three measures are not computed against any ground truth flow field and are therefore
unbiased (or better less biased) than the measure of errors in the previous experiments.

"""
import os
from os.path import join as jph
import tabulate
import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sympy.core.cache import clear_cache

from nilabels.tools.aux_methods.utils import print_and_run
from nilabels.tools.aux_methods.utils_nib import set_new_data

from calie.fields import queries as qr
from calie.fields import coordinate as coord

from benchmarking.a_main_controller import methods, spline_interpolation_order, steps, num_samples, bw_subjects, \
    ad_subjects
from benchmarking.b_path_manager import pfo_output_A4_SE2, pfo_output_A4_GL2, pfo_output_A4_HOM, \
    pfo_output_A4_GAU, pfo_output_A4_BW, pfo_output_A4_AD


def three_assessments_collector(control, verbose):

    # ----------------------- #
    # Retrieve data set paths
    # ----------------------- #

    if control['svf_dataset'].lower() in {'rotation', 'rotations'}:
        pfi_svf_list = [jph(pfo_output_A4_SE2, 'se2-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'linear'}:
        pfi_svf_list = [jph(pfo_output_A4_GL2, 'gl2-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'homography', 'homographies'}:
        pfi_svf_list = [jph(pfo_output_A4_HOM, 'hom-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'gauss'}:
        pfi_svf_list = [jph(pfo_output_A4_GAU, 'gau-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'brainweb'}:
        pfi_svf_list = [jph(pfo_output_A4_BW, 'bw-{}-algebra.npy'.format(sj)) for sj in bw_subjects]

    elif control['svf_dataset'].lower() in {'adni'}:
        pfi_svf_list = [jph(pfo_output_A4_AD, 'ad-{}-algebra.npy'.format(sj)) for sj in ad_subjects]
    else:
        raise IOError('Svf data set not given'.format(control['svf_dataset']))

    for pfi in pfi_svf_list:
        assert os.path.exists(pfi)

    if control['collect']:

        print('---------------------------------------------------------------------------')
        print('Collecting {} dataset  : bw-<s>-<algebra/group>.npy sj in BrainWeb '.format('computation'))
        print('---------------------------------------------------------------------------')

        pass

    if control['show']:
        pass


if __name__ == '__main__':

    clear_cache()

    # controller

    control_ = {'svf_dataset'   : 'rotation',  # can be rotation, linear, homography, gauss, brainweb, adni
                'computation'   : 'IC',  # can be IC, SA, SE
                'collect'       : True,
                'show'          : True}

    verbose_ = 1
    three_assessments_collector(control_, verbose_)