import copy
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from VECtorsToolkit.fields import queries as qr
from VECtorsToolkit.fields import generate_identities as gen_id


def triptych_image_quiver_image(image_1,
                                vector_field,
                                image_2,
                                fig_tag=5,
                                input_fig_size=(15, 5),
                                window_title_input='triptych',
                                interval_svf=1):

    fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    ax_1 = plt.subplot(131)
    ax_1.imshow(image_1, cmap='Greys',  interpolation='nearest', origin='lower')
    ax_1.axes.xaxis.set_ticklabels([])
    ax_1.axes.yaxis.set_ticklabels([])
    ax_1.set_xlabel('(grid)', fontdict=font)
    ax_1.set_aspect('equal')

    ax_2 = plt.subplot(132)
    x, y = np.meshgrid(np.arange(vector_field.shape[0]), np.arange(vector_field.shape[1]))
    ax_2.quiver(y[::interval_svf, ::interval_svf],
                x[::interval_svf, ::interval_svf],
                vector_field[::interval_svf, ::interval_svf, 0, 0, 0],
                vector_field[::interval_svf, ::interval_svf, 0, 0, 1], scale=1, scale_units='xy')
    ax_2.axes.xaxis.set_ticklabels([])
    ax_2.axes.yaxis.set_ticklabels([])
    ax_2.set_xlabel('(transformation)', fontdict=font)
    ax_2.set_aspect('equal')

    ax_3 = plt.subplot(133)
    ax_3.imshow(image_2, cmap='Greys',  interpolation='none', origin='lower')
    ax_3.axes.xaxis.set_ticklabels([])
    ax_3.axes.yaxis.set_ticklabels([])
    ax_3.set_xlabel('(transformed grid)', fontdict=font)
    ax_3.set_aspect('equal')

    return fig
