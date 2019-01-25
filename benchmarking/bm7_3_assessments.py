"""
Module to compute the
+ Inverse Consistency (IC)
+ Scalar Associativity (SA)
+ Stepwise Error (SE)
for the selected data set, among the ones generated in the previous experiments bm1_ to bm6_ .
These three measures are not computed against any ground truth flow field and are therefore
unbiased (or better less biased) than the measure of errors in the previous experiments.

"""
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

from benchmarking.a_main_controller import methods, spline_interpolation_order, steps, num_samples
from benchmarking.b_path_manager import pfo_output_A4_SE2, pfo_output_A4_GL2, pfo_output_A4_HOM, \
    pfo_output_A4_GAU, pfo_output_A4_BW, pfo_output_A4_AD


if __name__ == '__main__':

    clear_cache()

    # controller

    control = {'svf_dataset'   : 'rotation',  # can be rotation, translation, linear, homography, gauss, brainweb, adni
               'compute'       : 'IC',  # can be IC, SA, SE
               'show'          : True}

    bw_subjects   = []
    adni_subjects = []

    verbose = 1

    # ----------------------- #
    # Retrieve data set paths
    # ----------------------- #

    data_paths = []

    if control['svf_dataset'].lower() in {'rotation', 'rotations'}:
        pfi_svf_list = [jph(pfo_output_A4_SE2, 'se2-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'linear'}:
        pfi_svf_list = [jph(pfo_output_A4_GL2, 'gl2-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'homography', 'homographies'}:
        pfi_svf_list = [jph(pfo_output_A4_HOM, 'hom-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'gauss'}:
        pfi_svf_list = [jph(pfo_output_A4_HOM, 'gau-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'brainweb'}:
        pass

    elif control['svf_dataset'].lower() in {'adni'}:
        pass

    else:
        raise IOError('Svf data set not given'.format(control['svf_dataset']))




