"""
Visualiser for multiple vector field module.
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from calie.fields import queries as qr
from calie.fields import generate_identities as gen_id


font_top = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
font_bl = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
legend_prop = {'size': 11}

sns.set_style()


def see_2_fields_separate_and_overlay(input_vf_0, input_vf_1,
                                      anatomical_plane='axial',
                                      h_slice_0=0, h_slice_1=0,
                                      sample_0=(1, 1), sample_1=(1, 1),
                                      window_title_input='quiver 2 screens',
                                      input_color_0='b', input_color_1='r',
                                      title_input_0='Vector field 1', title_input_1= 'Vector field 2',
                                      title_input_both='Overlay',
                                      long_title_0=False,
                                      long_title_1=False,
                                      long_title_both=False,
                                      fig_tag=1,
                                      scale_0=1,
                                      scale_1=1,
                                      subtract_id_0=False, subtract_id_1=False):

    assert qr.check_is_vf(input_vf_0) == 2
    assert qr.check_is_vf(input_vf_1) == 2

    id_field_0 = gen_id.id_eulerian_like(input_vf_0)  # other option is casting with Field()
    id_field_1 = gen_id.id_eulerian_like(input_vf_1)

    input_field_0 = copy.deepcopy(input_vf_0)
    input_field_1 = copy.deepcopy(input_vf_1)

    if subtract_id_0:
        input_field_0 -= id_field_0

    if subtract_id_1:
        input_field_1 -= id_field_1

    fig = plt.figure(fig_tag, figsize=(14, 5), dpi=80)
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.15)
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)

    fig.canvas.set_window_title(window_title_input)

    # anatomical plane is the same for both figures
    if anatomical_plane == 'axial':

        ax0.quiver(id_field_0[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
                   id_field_0[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
                   input_field_0[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
                   input_field_0[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
                   color=input_color_0, linewidths=0.2, units='xy', angles='xy', scale=scale_0, scale_units='xy')

        ax1.quiver(id_field_1[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
                   id_field_1[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
                   input_field_1[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
                   input_field_1[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
                   color=input_color_1, linewidths=0.2, units='xy', angles='xy', scale=scale_0, scale_units='xy')

        ax2.quiver(id_field_0[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
                   id_field_0[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
                   input_field_0[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
                   input_field_0[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
                   color=input_color_0, linewidths=0.2, units='xy', angles='xy', scale=scale_0, scale_units='xy')
        ax2.quiver(id_field_1[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
                   id_field_1[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
                   input_field_1[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
                   input_field_1[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
                   color=input_color_1, linewidths=0.2, units='xy', angles='xy', scale=scale_0, scale_units='xy')

    elif anatomical_plane == 'sagittal':

        ax0.quiver(id_field_0[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
                   id_field_0[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
                   input_field_0[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
                   input_field_0[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
                   color=input_color_0, linewidths=0.2, units='xy', angles='xy', scale=scale_0, scale_units='xy')

        ax1.quiver(id_field_1[::sample_1[0], h_slice_1, ::sample_1[1], 0, 0],
                   id_field_1[::sample_1[0], h_slice_1, ::sample_1[1], 0, 1],
                   input_field_1[::sample_1[0], h_slice_1, ::sample_1[1], 0, 0],
                   input_field_1[::sample_1[0], h_slice_1, ::sample_1[1], 0, 1],
                   color=input_color_1, linewidths=0.2, units='xy', angles='xy', scale=scale_1, scale_units='xy')

        ax2.quiver(id_field_0[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
                   id_field_0[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
                   input_field_0[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
                   input_field_0[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
                   color=input_color_0, linewidths=0.2, units='xy', angles='xy', scale=scale_0, scale_units='xy')
        ax2.quiver(id_field_1[::sample_1[0], h_slice_1, ::sample_1[1], 0, 0],
                   id_field_1[::sample_1[0], h_slice_1, ::sample_1[1], 0, 1],
                   input_field_1[::sample_1[0], h_slice_1, ::sample_1[1], 0, 0],
                   input_field_1[::sample_1[0], h_slice_1, ::sample_1[1], 0, 1],
                   color=input_color_1, linewidths=0.2, units='xy', angles='xy', scale=scale_1, scale_units='xy')

    elif anatomical_plane == 'coronal':

        ax0.quiver(id_field_0[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
                   id_field_0[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
                   input_field_0[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
                   input_field_0[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
                   color=input_color_0, linewidths=0.2, units='xy', angles='xy', scale=scale_0, scale_units='xy')

        ax1.quiver(id_field_1[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
                   id_field_1[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
                   input_field_1[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
                   input_field_1[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
                   color=input_color_1, linewidths=0.2, units='xy', angles='xy', scale=scale_1, scale_units='xy')

        ax2.quiver(id_field_0[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
                   id_field_0[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
                   input_field_0[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
                   input_field_0[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
                   color=input_color_0, linewidths=0.2, units='xy', angles='xy', scale=scale_0, scale_units='xy')
        ax2.quiver(id_field_1[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
                   id_field_1[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
                   input_field_1[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
                   input_field_1[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
                   color=input_color_1, linewidths=0.2, units='xy', angles='xy', scale=scale_1, scale_units='xy')

    else:
        raise TypeError('anatomical_plane_0 must be axial, sagittal or coronal')

    if long_title_0:
        ax0.set_title(title_input_0 + ', ' + str(anatomical_plane) + ' plane, slice ' + str(h_slice_0))
    else:
        ax0.set_title(title_input_0)

    if long_title_1:
        ax1.set_title(title_input_1 + ', ' + str(anatomical_plane) + ' plane, slice ' + str(h_slice_1))
    else:
        ax1.set_title(title_input_1)

    if long_title_both:
        ax2.set_title(title_input_both + ', in the same window')
    else:
        ax2.set_title(title_input_both)

    ax0.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax0.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax0.set_axisbelow(True)

    ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.set_axisbelow(True)

    ax2.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax2.set_axisbelow(True)

    fig.set_tight_layout(True)


def see_n_fields_separate(list_of_vf,
                          row_fig=2,
                          col_fig=5,
                          input_figsize=(15, 6),
                          anatomical_plane='axial',
                          h_slice=0,
                          sample=(1, 1),
                          window_title_input='quiver',
                          title_input=None,
                          fig_tag=1,
                          scale=1,
                          xy_lims=None,
                          set_aspect_equal=True,
                          subtract_id=None,
                          input_color=None,
                          linewidths=0.01,
                          width=0.03):
    """
    :param list_of_vf:
    :param row_fig:
    :param col_fig:
    :param input_figsize:
    :param anatomical_plane:
    :param h_slice:
    :param sample:
    :param window_title_input:
    :param title_input:
    :param fig_tag:
    :param scale:
    :param xy_lims: (x0, x1, y0, y1) limits of the axis
    :param set_aspect_equal: set equals ratio of x and y axes
    :param subtract_id:
    :param input_color:
    :param linewidths:
    :param width:
    :return:
    """
    # TODO: input sanity check

    n = len(list_of_vf)

    if title_input is None:
        title_input = ('2d vector field',) * n
    if subtract_id is None:
        subtract_id = (False,) * 10
    if input_color is None:
        input_color = ('r', 'b', 'g', 'c', 'm', 'k') * (n / 6 + n % 6)

    fig = plt.figure(fig_tag, figsize=input_figsize, dpi=80)
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.15)

    fig.canvas.set_window_title(window_title_input)

    for num_vf, input_vf in enumerate(list_of_vf):

        # add subplot here:
        ax  = fig.add_subplot(row_fig, col_fig, num_vf+1)

        if input_vf is not None:

            assert qr.check_is_vf(input_vf) == 2

            id_field = gen_id.id_eulerian_like(input_vf)
            input_field_copy = copy.deepcopy(input_vf)

            if subtract_id[num_vf]:
                input_field_copy -= id_field

            # anatomical plane and h axis is (at the moment) the same for every picture!
            if anatomical_plane == 'axial':
                ax.quiver(id_field[::sample[0], ::sample[1], h_slice, 0, 0],
                               id_field[::sample[0], ::sample[1], h_slice, 0, 1],
                               input_field_copy[::sample[0], ::sample[1], h_slice, 0, 0],
                               input_field_copy[::sample[0], ::sample[1], h_slice, 0, 1],
                               color=input_color[num_vf], linewidths=linewidths, width=width, scale=scale, scale_units='xy', units='xy', angles='xy', )

            elif anatomical_plane == 'sagittal':
                ax.quiver(id_field[::sample[0], h_slice, ::sample[1], 0, 0],
                               id_field[::sample[0], h_slice, ::sample[1], 0, 1],
                               input_field_copy[::sample[0], h_slice, ::sample[1], 0, 0],
                               input_field_copy[::sample[0], h_slice, ::sample[1], 0, 1],
                               color=input_color[num_vf], linewidths=1, units='xy', angles='xy', scale=scale, scale_units='xy')

            elif anatomical_plane == 'coronal':
                ax.quiver(id_field[h_slice, ::sample[0], ::sample[1], 0, 0],
                           id_field[h_slice, ::sample[0], ::sample[1], 0, 1],
                           input_field_copy[h_slice, ::sample[0], ::sample[1], 0, 0],
                           input_field_copy[h_slice, ::sample[0], ::sample[1], 0, 1],
                           color=input_color[num_vf], linewidths=1, units='xy', angles='xy', scale=scale, scale_units='xy')
            else:
                raise TypeError('Anatomical_plane must be axial, sagittal or coronal')

            ax.set_title(title_input[num_vf], fontdict=font_top)

            if xy_lims is not None:
                ax.set_xlim(xy_lims[0], xy_lims[1])
                ax.set_ylim(xy_lims[2], xy_lims[3])

            if set_aspect_equal:
                ax.set_aspect('equal')

            ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            ax.set_axisbelow(True)

    fig.set_tight_layout(True)


def see_n_fields_special(list_of_list_vf,
                         row_fig=2,
                         col_fig=5,
                         anatomical_plane='axial',
                         h_slice=0,
                         sample=(1, 1),
                         window_title_input='quiver',
                         titles_input=None,
                         input_figsize=(14, 5.5),
                         zoom_input=None,
                         fig_tag=1,
                         scale=1,
                         subtract_id=None,
                         colors_input=None,
                         labels_input=None,
                         legend_on=False):
    """

    :param list_of_list_vf:
    :param row_fig:
    :param col_fig:
    :param anatomical_plane:
    :param h_slice:
    :param sample:
    :param window_title_input:
    :param titles_input:
    :param input_figsize:
    :param zoom_input:
    :param fig_tag:
    :param scale:
    :param subtract_id:
    :param colors_input:
    :param labels_input:
    :param legend_on:
    :return:
    """

    n     = len(list_of_list_vf)  # number of subplot
    num_v = [len(vect) for vect in list_of_list_vf]

    if titles_input is None:
        titles_input = ('2d vector fields',) * n
    if subtract_id is None:
        subtract_id = (False,) * n * sum(num_v)
    if colors_input is None:
        colors_input = ('r', 'b', 'g', 'c', 'm', 'k') * (n / 6 + n % 6)
    if zoom_input is None:
        # zoom input is defined as [x_0, x_1, y_0, y_1] as the extreme of the input
        x_0, y_0 = 0, 0
        x_1, y_1 = 10, 10
    else:
        x_0, x_1, y_0, y_1 = zoom_input
    if labels_input is None:
        labels_input = ['legend'] * n * sum(num_v)

    fig = plt.figure(fig_tag, figsize=input_figsize, dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)

    fig.canvas.set_window_title(window_title_input)

    for num_list_of_obj, input_list_of_obj in enumerate(list_of_list_vf):

        # add subplot here:
        ax  = fig.add_subplot(row_fig, col_fig, num_list_of_obj + 1)

        if input_list_of_obj is not None:

            for num_vf, input_vf in enumerate(input_list_of_obj):

                main_index = int(np.sum(num_v[:num_list_of_obj]) + num_vf)

                assert qr.check_is_vf(input_vf) == 2

                id_field = gen_id.id_eulerian_like(input_vf)
                input_field_copy = copy.deepcopy(input_vf)

                if subtract_id[num_vf]:
                    input_field_copy -= id_field

                # anatomical plane and h axis is (at the moment) the same for every picture!
                if anatomical_plane == 'axial':
                    q = ax.quiver(id_field[x_0:x_1:sample[0], y_0:y_1:sample[1], h_slice, 0, 0],
                                  id_field[x_0:x_1:sample[0], y_0:y_1:sample[1], h_slice, 0, 1],
                                  input_field_copy[x_0:x_1:sample[0], y_0:y_1:sample[1], h_slice, 0, 0],
                                  input_field_copy[x_0:x_1:sample[0], y_0:y_1:sample[1], h_slice, 0, 1],
                                  color=colors_input[main_index],
                                  label=labels_input[main_index],
                                  linewidths=0.01, width=0.03, scale=scale, scale_units='xy',
                                  units='xy', angles='xy', )

                elif anatomical_plane == 'sagittal':
                    q = ax.quiver(id_field[x_0:x_1:sample[0], h_slice, y_0:y_1:sample[1], 0, 0],
                                  id_field[::sample[0], h_slice, ::sample[1], 0, 1],
                                  input_field_copy[x_0:x_1:sample[0], h_slice, y_0:y_1:sample[1], 0, 0],
                                  input_field_copy[x_0:x_1:sample[0], h_slice, y_0:y_1:sample[1], 0, 1],
                                  color=colors_input[main_index],
                                  label=labels_input[main_index],
                                  linewidths=1, units='xy', angles='xy', scale=scale,
                                  scale_units='xy')

                elif anatomical_plane == 'coronal':
                    q = ax.quiver(id_field[h_slice, x_0:x_1:sample[0], y_0:y_1:sample[1], 0, 0],
                                  id_field[h_slice, x_0:x_1:sample[0], y_0:y_1:sample[1], 0, 1],
                                  input_field_copy[h_slice, x_0:x_1:sample[0], y_0:y_1:sample[1], 0, 0],
                                  input_field_copy[h_slice, x_0:x_1:sample[0], y_0:y_1:sample[1], 0, 1],
                                  color=colors_input[main_index],
                                  label=labels_input[main_index],
                                  linewidths=1, units='xy', angles='xy', scale=scale,
                                  scale_units='xy')
                else:
                    raise TypeError('Anatomical_plane must be axial, sagittal or coronal')

                if legend_on:
                    qk = ax.quiverkey(q, 2, 2, 3, labels_input[num_list_of_obj], coordinates='data', labelcolor='k',
                                      fontproperties={'weight': 'bold', 'size': 8})
                    # qk.text.set_backgroundcolor('w')

            ax.set_title(titles_input[num_list_of_obj])

            ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            ax.set_axisbelow(True)

    fig.set_tight_layout(False)


def see_overlay_of_n_fields(list_of_vf,
                            sample=(1, 1),
                            window_title_input='quiver',
                            title_input='2d vector fields',
                            fig_tag=1, scale=1,
                            input_color=('r', 'b'),
                            subtract_id=None):

    fig = plt.figure(fig_tag)
    ax0 = fig.add_subplot(111)
    fig.canvas.set_window_title(window_title_input)

    for num_vf, input_vf in enumerate(list_of_vf):

        if input_vf is not None:

            id_field = gen_id.id_eulerian_like(input_vf)
            input_field = copy.deepcopy(input_vf)

            assert qr.check_is_vf(input_vf) == 2

            if subtract_id is not None:
                if subtract_id[num_vf]:
                    input_field -= id_field

            # figure 1
            ax0.quiver(id_field[::sample[0], ::sample[1], 0, 0, 0],
                       id_field[::sample[0], ::sample[1], 0, 0, 1],
                       input_field[::sample[0], ::sample[1], 0, 0, 0],
                       input_field[::sample[0], ::sample[1], 0, 0, 1],
                       color=input_color[num_vf], linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')

        ax0.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax0.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax0.set_axisbelow(True)

        ax0.set_title(title_input)

    fig.set_tight_layout(True)


def see_overlay_of_n_fields_3dd(list_of_obj,
                                anatomical_plane='axial',
                                h_slice=0, sample=(1, 1),
                                window_title_input='quiver',
                                title_input='2d vector fields',
                                long_title=False,
                                fig_tag=1, scale=1,
                                input_color=('r', 'b'),
                                input_label=None,
                                subtract_id=None):
    """
    NOT TESTED!
    :param list_of_obj:
    :param anatomical_plane:
    :param h_slice:
    :param sample:
    :param window_title_input:
    :param title_input:
    :param long_title:
    :param fig_tag:
    :param scale:
    :param input_color:
    :param input_label:
    :param subtract_id:
    :return:
    """

    fig = plt.figure(fig_tag)
    ax0 = fig.add_subplot(111)
    fig.canvas.set_window_title(window_title_input)

    for num_obj, input_obj in enumerate(list_of_obj):

        if input_obj is not None:

            id_field = input_obj.__class__.generate_id_from_obj(input_obj)
            input_field = copy.deepcopy(input_obj)

            if not len(input_obj.shape) == 5:
                raise TypeError('Wrong input size for a the field' + str(input_obj))

            if subtract_id is not None:
                if subtract_id[num_obj]:
                    input_field -= id_field

            # figure 1
            if anatomical_plane == 'axial':
                ax0.quiver(id_field[::sample[0], ::sample[1], h_slice, 0, 0],
                           id_field[::sample[0], ::sample[1], h_slice, 0, 1],
                           input_field[::sample[0], ::sample[1], h_slice, 0, 0],
                           input_field[::sample[0], ::sample[1], h_slice, 0, 1],
                           color=input_color[num_obj], linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')

            elif anatomical_plane == 'sagittal':
                ax0.quiver(id_field[::sample[0], h_slice, ::sample[1], 0, 0],
                           id_field[::sample[0], h_slice, ::sample[1], 0, 1],
                           input_field[::sample[0], h_slice, ::sample[1], 0, 0],
                           input_field[::sample[0], h_slice, ::sample[1], 0, 1],
                           color=input_color[num_obj], linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')

            elif anatomical_plane == 'coronal':
                ax0.quiver(id_field[h_slice, ::sample[0], ::sample[1], 0, 0],
                           id_field[h_slice, ::sample[0], ::sample[1], 0, 1],
                           input_field[h_slice, ::sample[0], ::sample[1], 0, 0],
                           input_field[h_slice, ::sample[0], ::sample[1], 0, 1],
                           color=input_color[num_obj], linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')
            else:
                raise TypeError('anatomical_plane_1 must be axial, sagittal or coronal')

        ax0.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax0.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax0.set_axisbelow(True)

        if long_title:
            ax0.set_title(title_input + ', ' + str(anatomical_plane) + ' plane, slice ' + str(h_slice))
        else:
            ax0.set_title(title_input)

    fig.set_tight_layout(True)