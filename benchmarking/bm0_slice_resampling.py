import os
import numpy as np

from .b_path_manager import pfo_brainweb


def run_experiment(control, param):

    subject_id = 'BW38'
    labels_brain = [1, 2, 3]
    y_slice = 118

    if control['prepare_data']:
        # skull_strip subject
        pass
        # save transformation

    if control['elaborate']:

        pass

    if control['save_results']:
        pass

    if control['show_results']:
        pass


if __name__ == '__main__':
    controller = {'prepare_data' : True,
                  'elaborate'    : True,
                  'save_results' : True,
                  'show_results' : True}
    parameters = {}

    run_experiment(controller, parameters)
