import copy

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from calie.fields import queries as qr
from calie.fields import generate_identities as gen_id


def see_array(in_array, extra_image=None, scale=None, num_fig=1):

    fig = plt.figure(num_fig, figsize=(6, 7.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_position([0.1, 0.29, 0.8, 0.7])

    fig.canvas.set_window_title('Image in matrix coordinates, C convention.')

    dims = in_array.shape  # (i,j,k,t,d)
    dims_mean = [int(d / 2) for d in dims]

    init_ax = 0
    axcolor = '#ababab'

    l = ax.imshow(in_array.take(dims_mean[init_ax], axis=init_ax), aspect='equal', origin='lower', interpolation='nearest', cmap='gray')
    #dot = ax.plot(dims_mean[1], dims_mean[2], 'r+')

    i_slider_plot = plt.axes([0.25, 0.2, 0.65, 0.03], axisbg='r')
    i_slider = Slider(i_slider_plot, 'i', 0, dims[0] - 1, valinit=dims_mean[0], valfmt='%1i')

    j_slider_plot = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg='g')
    j_slider = Slider(j_slider_plot, 'j', 0, dims[1] - 1, valinit=dims_mean[1], valfmt='%1i')

    k_slider_plot = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg='b')
    k_slider = Slider(k_slider_plot, 'k', 0, dims[2] - 1, valinit=dims_mean[2], valfmt='%1i')

    axis_selector_plot = plt.axes([0.02, 0.1, 0.15, 0.13], axisbg=axcolor)
    axis_selector = RadioButtons(axis_selector_plot, ('jk', 'ik', 'ij'), active=0)

    center_image_button_plot = plt.axes([0.02, 0.04, 0.15, 0.04])
    center_image_button = Button(center_image_button_plot, 'Center', color=axcolor, hovercolor='0.975')

    def update_plane(label):

        global l

        if label == 'jk':
            new_i = int(i_slider.val)
            l = ax.imshow(in_array.take(new_i, axis=0), aspect='equal', origin='lower', interpolation='nearest', cmap='gray')
            ax.set_xlim([0, dims[2]])
            ax.set_ylim([0, dims[1]])
            #l.set_array(in_array.take(new_i, axis=0))

        if label == 'ik':
            new_j = int(j_slider.val)
            l = ax.imshow(in_array.take(new_j, axis=1), aspect='equal', origin='lower', interpolation='nearest', cmap='gray')

            ax.set_xlim([0, dims[2]])
            ax.set_ylim([0, dims[0]])
            #l.set_array(in_array.take(new_j, axis=1))

        if label == 'ij':
            new_k = int(k_slider.val)
            l = ax.imshow(in_array.take(new_k, axis=2), aspect='equal', origin='lower', interpolation='nearest', cmap='gray')
            ax.set_xlim([0, dims[1]])
            ax.set_ylim([0, dims[0]])
            #l.set_array(in_array.take(new_k, axis=2))

        fig.canvas.draw()

    def update_slides(val):

        global l

        new_i = int(i_slider.val)
        new_j = int(j_slider.val)
        new_k = int(k_slider.val)

        if axis_selector.value_selected == 'jk':
            l.set_array(in_array.take(new_i, axis=0))
        if axis_selector.value_selected == 'ik':
            l.set_array(in_array.take(new_j, axis=1))
        if axis_selector.value_selected == 'ij':
            l.set_array(in_array.take(new_k, axis=2))

        fig.canvas.draw_idle()

    def reset_slides(event):
        i_slider.reset()
        j_slider.reset()
        k_slider.reset()

    axis_selector.on_clicked(update_plane)

    i_slider.on_changed(update_slides)
    j_slider.on_changed(update_slides)
    k_slider.on_changed(update_slides)

    center_image_button.on_clicked(reset_slides)

    '''
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print 'vx = %d, vy = %d' % (ix, iy)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    '''

    if len(dims) >= 4:

        t_slider_plot = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg='b')
        t_slider = Slider(t_slider_plot, 'k', 0, dims[3], valinit=dims_mean[init_ax], valfmt='%1i')

        def update_t(val):
            new_t = int(t_slider.val)
            l.set_array(in_array.take(new_t, axis=3))
            fig.canvas.draw_idle()

        t_slider.on_changed(update_t)

    plt.show()


def see_one_slice(input_vf,
                  anatomical_plane='axial',
                  h_slice=0,
                  sample=(1, 1),
                  window_title_input='quiver',
                  title_input='2d vector field',
                  long_title=False,
                  fig_tag=1, scale=1,
                  subtract_id=False,
                  input_color='b',
                  annotate=None,
                  annotate_position=(1, 1)):
    qr.check_is_vf(input_vf)
    d = input_vf.shape[-1]
    if not d == 2:
            raise TypeError('See field 2d works only for 2d to 2d fields.')

    id_field = gen_id.id_lagrangian(list(input_vf.shape[:d]))

    fig = plt.figure(fig_tag)
    ax0 = fig.add_subplot(111)
    fig.canvas.set_window_title(window_title_input)

    input_field_copy = copy.deepcopy(input_vf)

    if subtract_id:
        input_field_copy -= id_field

    if anatomical_plane == 'axial':
        ax0.quiver(id_field[::sample[0], ::sample[1], h_slice, 0, 0],
                   id_field[::sample[0], ::sample[1], h_slice, 0, 1],
                   input_field_copy[::sample[0], ::sample[1], h_slice, 0, 0],
                   input_field_copy[::sample[0], ::sample[1], h_slice, 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, scale=scale, scale_units='xy', units='xy',
                   angles='xy')
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')

    elif anatomical_plane == 'sagittal':
        ax0.quiver(id_field[::sample[0], h_slice, ::sample[1], 0, 0],
                   id_field[::sample[0], h_slice, ::sample[1], 0, 1],
                   input_field_copy[::sample[0], h_slice, ::sample[1], 0, 0],
                   input_field_copy[::sample[0], h_slice, ::sample[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale,
                   scale_units='xy')

    elif anatomical_plane == 'coronal':
        ax0.quiver(id_field[h_slice, ::sample[0], ::sample[1], 0, 0],
                   id_field[h_slice, ::sample[0], ::sample[1], 0, 1],
                   input_field_copy[h_slice, ::sample[0], ::sample[1], 0, 0],
                   input_field_copy[h_slice, ::sample[0], ::sample[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale,
                   scale_units='xy')
    else:
        raise TypeError('Anatomical_plane must be axial, sagittal or coronal')

    ax0.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax0.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax0.set_axisbelow(True)

    if long_title:
        ax0.set_title(title_input + ', ' + str(anatomical_plane) + ' plane, slice ' + str(h_slice))
    else:
        ax0.set_title(title_input)

    if annotate is not None:
        ax0.text(annotate_position[0], annotate_position[1], annotate)

    plt.axes().set_aspect('equal', 'datalim')

    return fig


def see_field(input_vf,
              anatomical_plane='axial',
              h_slice=0, sample=(1, 1),
              window_title_input='quiver',
              title_input='2d vector field',
              long_title=False,
              fig_tag=1,
              scale=1,
              subtract_id=False,
              input_color='b',
              annotate=None, annotate_position=(1, 1)):

    qr.check_is_vf(input_vf)

    id_field = gen_id.id_eulerian_like(input_vf)

    fig = plt.figure(fig_tag)
    ax0 = fig.add_subplot(111)
    fig.canvas.set_window_title(window_title_input)

    input_field_copy = copy.deepcopy(input_vf)

    if subtract_id:
        input_field_copy -= id_field

    if anatomical_plane == 'axial':
        ax0.quiver(id_field[::sample[0], ::sample[1], h_slice, 0, 0],
                   id_field[::sample[0], ::sample[1], h_slice, 0, 1],
                   input_field_copy[::sample[0], ::sample[1], h_slice, 0, 0],
                   input_field_copy[::sample[0], ::sample[1], h_slice, 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, scale=scale, scale_units='xy', units='xy',
                   angles='xy')
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')

    elif anatomical_plane == 'sagittal':
        ax0.quiver(id_field[::sample[0], h_slice, ::sample[1], 0, 0],
                   id_field[::sample[0], h_slice, ::sample[1], 0, 1],
                   input_field_copy[::sample[0], h_slice, ::sample[1], 0, 0],
                   input_field_copy[::sample[0], h_slice, ::sample[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale,
                   scale_units='xy')

    elif anatomical_plane == 'coronal':
        ax0.quiver(id_field[h_slice, ::sample[0], ::sample[1], 0, 0],
                   id_field[h_slice, ::sample[0], ::sample[1], 0, 1],
                   input_field_copy[h_slice, ::sample[0], ::sample[1], 0, 0],
                   input_field_copy[h_slice, ::sample[0], ::sample[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale,
                   scale_units='xy')
    else:
        raise TypeError('Anatomical_plane must be axial, sagittal or coronal')

    ax0.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax0.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax0.set_axisbelow(True)

    if long_title:
        ax0.set_title(title_input + ', ' + str(anatomical_plane) + ' plane, slice ' + str(h_slice))
    else:
        ax0.set_title(title_input)

    if annotate is not None:
        ax0.text(annotate_position[0], annotate_position[1], annotate)

    plt.axes().set_aspect('equal', 'datalim')

    return fig


def see_2_fields(input_obj_0, input_obj_1,
                 anatomical_plane_0='axial', anatomical_plane_1='axial',
                 h_slice_0=0, h_slice_1=0,
                 sample_0=(1, 1), sample_1=(1, 1),
                 window_title_input='quiver 2 screens',
                 title_input_0='Vector field', title_input_1='Vector field',
                 long_title_0=False, long_title_1=False,
                 fig_tag=1, scale_0=1, scale_1=1,
                 subtract_id_0=False, subtract_id_1=False, input_color='b'):
    """

    :param input_obj_0:
    :param input_obj_1:
    :param anatomical_plane_0:
    :param anatomical_plane_1:
    :param h_slice_0:
    :param h_slice_1:
    :param sample_0:
    :param sample_1:
    :param window_title_input:
    :param title_input_0:
    :param title_input_1:
    :param long_title_0:
    :param long_title_1:
    :param fig_tag:
    :param scale_0:
    :param scale_1:
    :param subtract_id_0:
    :param subtract_id_1:
    :param input_color:
    :return:
    """

    qr.check_is_vf(input_obj_0)
    qr.check_is_vf(input_obj_1)

    id_field_0 = gen_id.id_eulerian_like(input_obj_0)  # other option is casting with Field()
    id_field_1 = gen_id.id_eulerian_like(input_obj_1)

    input_field_0 = copy.deepcopy(input_obj_0)
    input_field_1 = copy.deepcopy(input_obj_1)

    if subtract_id_0:
        input_field_0 -= id_field_0

    if subtract_id_1:
        input_field_1 -= id_field_1

    fig = plt.figure(fig_tag)
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    fig.canvas.set_window_title(window_title_input)

    # figure 0
    if anatomical_plane_0 == 'axial':
        ax0.quiver(id_field_0[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
                   id_field_0[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
                   input_field_0[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
                   input_field_0[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0,
                   scale_units='xy')

    elif anatomical_plane_0 == 'sagittal':
        ax0.quiver(id_field_0[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
                   id_field_0[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
                   input_field_0[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
                   input_field_0[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0,
                   scale_units='xy')

    elif anatomical_plane_0 == 'coronal':
        ax0.quiver(id_field_0[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
                   id_field_0[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
                   input_field_0[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
                   input_field_0[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0,
                   scale_units='xy')
    else:
        raise TypeError('anatomical_plane_0 must be axial, sagittal or coronal')

    if long_title_0:
        ax0.set_title(title_input_0 + ', ' + str(anatomical_plane_0) + ' plane, slice ' + str(h_slice_0))
    else:
        ax0.set_title(title_input_0)

    ax0.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax0.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax0.set_axisbelow(True)

    # figure 1
    if anatomical_plane_1 == 'axial':
        ax1.quiver(id_field_1[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
                   id_field_1[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
                   input_field_1[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
                   input_field_1[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1,
                   scale_units='xy')

    elif anatomical_plane_1 == 'sagittal':
        ax1.quiver(id_field_0[::sample_1[0], h_slice_0, ::sample_1[1], 0, 0],
                   id_field_0[::sample_1[0], h_slice_0, ::sample_1[1], 0, 1],
                   input_field_0[::sample_1[0], h_slice_0, ::sample_1[1], 0, 0],
                   input_field_0[::sample_1[0], h_slice_0, ::sample_1[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1,
                   scale_units='xy')

    elif anatomical_plane_1 == 'coronal':
        ax1.quiver(id_field_1[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
                   id_field_1[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
                   input_field_1[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
                   input_field_1[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
                   color=input_color, linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1,
                   scale_units='xy')
    else:
        raise TypeError('anatomical_plane_1 must be axial, sagittal or coronal')

    if long_title_1:
        ax1.set_title(title_input_1 + ', ' + str(anatomical_plane_1) + ' plane, slice ' + str(h_slice_1))
    else:
        ax1.set_title(title_input_1)

    ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.set_axisbelow(True)

    fig.set_tight_layout(True)


def quiver_3d(svf, flow=None, sample=(1, 1, 1), scale=1):

    omega_svf  = qr.check_is_vf(svf)
    omega_flow = qr.check_is_vf(flow)

    np.testing.assert_array_equal(omega_flow, omega_svf)

    id_field = gen_id.id_eulerian_like(svf)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = id_field[::sample[0], ::sample[1], ::sample[2], 0, 0]
    y = id_field[::sample[0], ::sample[1], ::sample[2], 0, 1]
    z = id_field[::sample[0], ::sample[1], ::sample[2], 0, 2]

    x = x.reshape(np.product(x.shape))
    y = y.reshape(np.product(y.shape))
    z = z.reshape(np.product(z.shape))

    svf_x = svf[::sample[0], ::sample[1], ::sample[2], 0, 0]
    svf_y = svf[::sample[0], ::sample[1], ::sample[2], 0, 1]
    svf_z = svf[::sample[0], ::sample[1], ::sample[2], 0, 2]

    svf_x = svf_x.reshape(np.product(svf_x.shape))
    svf_y = svf_y.reshape(np.product(svf_y.shape))
    svf_z = svf_z.reshape(np.product(svf_z.shape))

    lengths = scale * np.sqrt(svf_x ** 2 + svf_y ** 2 + svf_z ** 2)

    for x1, y1, z1, u1, v1, w1, l in zip(x, y, z, svf_x, svf_y, svf_z, lengths):
        ax.quiver(x1, y1, z1, u1, v1, w1, pivot='tail', length=l, color='r', linewidths=0.1)

    if flow is not None:
        flow_x = flow[::sample[0], ::sample[1], ::sample[2], 0, 0]
        flow_y = flow[::sample[0], ::sample[1], ::sample[2], 0, 1]
        flow_z = flow[::sample[0], ::sample[1], ::sample[2], 0, 2]

        flow_x = flow_x.reshape(np.product(flow_x.shape))
        flow_y = flow_y.reshape(np.product(flow_y.shape))
        flow_z = flow_z.reshape(np.product(flow_z.shape))

        lengthsflow = scale * np.sqrt(svf_x ** 2 + svf_y ** 2 + svf_z ** 2)

        for x1, y1, z1, u1, v1, w1, l in zip(x, y, z, flow_x, flow_y, flow_z, lengthsflow):
            ax.quiver(x1, y1, z1, u1, v1, w1, pivot='tail', length=l, color='b', linewidths=0.1)


if __name__ == '__main__':

    import scipy

    from calie.transformations import linear
    from calie.fields import generate as gen
    from calie.operations import lie_exp

    # --- Linear example

    beta = 0.1
    omega = (10, 12, 13)

    # generate matrix
    dm1 = beta * linear.randomgen_linear_by_taste(1, 1, [int(c / 2) for c in omega])
    m1 = scipy.linalg.expm(dm1)

    # generate SVF
    svf1 = gen.generate_from_matrix(omega, dm1, t=1, structure='algebra')
    flow1_ground = gen.generate_from_matrix(omega, m1, t=1, structure='group')

    quiver_3d(svf1, flow1_ground)

    plt.show(block=False)

    # --- Gaussian example

    # generate SVF
    svf1 = gen.generate_random(omega, 1, (1, 1))

    l_exp = lie_exp.LieExp()
    l_exp.s_i_o = 3
    flow1_ground = l_exp.gss_aei(svf1)

    quiver_3d(svf1, flow1_ground)

    plt.show(block=True)

    #
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # x, y, z = np.meshgrid(np.arange(-1, 1, 0.4),
    #                       np.arange(-1, 1, 0.4),
    #                       np.arange(-1, 1, 0.4))
    # x = x.reshape(np.product(x.shape))
    # y = y.reshape(np.product(y.shape))
    # z = z.reshape(np.product(z.shape))
    #
    # scale_ = 0.02
    # u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    # v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    # w = np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
    # lengths = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    #
    # for x1, y1, z1, u1, v1, w1, l in zip(x, y, z, u, v, w, lengths):
    #     ax.quiver(x1, y1, z1, u1, v1, w1, pivot='tail', length=l * 0.5)
    #
    # # ax.scatter(x, y, z, color='black')
    # plt.show()




# def see_field_subregion(input_obj,
#                         anatomical_plane='axial',
#                         h_slice=0, sample=(1, 1),
#                         subregion=([0, 20], [0, 20]),
#                         window_title_input='quiver',
#                         title_input= 'Vector field',
#                         long_title=False,
#                         fig_tag=1, scale=1,
#                         subtract_id=False):
#     """
#     Fields and children visualizer.
#     :param input_obj: print a slice of a 2d or 3d svf
#     :param anatomical_plane: if 2d is axial, with slice = 0
#     :param h_slice: the slice of the plane we want to plot
#     :param sample: does not plot all the vertical lines but a subspace
#     :param subregion: subregion of the initial domain.
#     :param long_title: add plane and slice information to the title
#     :param fig_tag: change the tag of the figure for each new instance!
#     The default will create problems for multiple images in the same module!
#     :return: plot the quiver of the svf in the sub-region we are interested in.
#     """
#     check_is_vector_field(input_vf)
#
#     id_field = input_obj.__class__.generate_id_from_obj(input_obj)
#
#     input_field_copy = copy.deepcopy(input_obj)
#
#     if subtract_id:
#         input_field_copy -= id_field
#
#     fig = plt.figure(fig_tag)
#     ax0 = fig.add_subplot(111)
#     fig.canvas.set_window_title(window_title_input)
#
#     if anatomical_plane == 'axial':
#         ax0.quiver(id_field.field[subregion[0][0]:subregion[0][1]:sample[0],
#                             subregion[1][0]:subregion[1][1]:sample[1],
#                             h_slice, 0, 0],
#                    id_field.field[subregion[0][0]:subregion[0][1]:sample[0],
#                             subregion[1][0]:subregion[1][1]:sample[1],
#                             h_slice, 0, 1],
#                    input_field_copy.field[subregion[0][0]:subregion[0][1]:sample[0],
#                                     subregion[1][0]:subregion[1][1]:sample[1],
#                                     h_slice, 0, 0],
#                    input_field_copy.field[subregion[0][0]:subregion[0][1]:sample[0],
#                                     subregion[1][0]:subregion[1][1]:sample[1],
#                                     h_slice, 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')
#
#     elif anatomical_plane == 'sagittal':
#         ax0.quiver(id_field.field[subregion[0][0]:subregion[0][1]:sample[0],
#                             h_slice,
#                             subregion[1][0]:subregion[1][1]:sample[1], 0, 0],
#                    id_field.field[subregion[0][0]:subregion[0][1]:sample[0],
#                             h_slice,
#                             subregion[1][0]:subregion[1][1]:sample[1], 0, 1],
#                    input_field_copy.field[subregion[0][0]:subregion[0][1]:sample[0],
#                                     h_slice,
#                                     subregion[1][0]:subregion[1][1]:sample[1], 0, 0],
#                    input_field_copy.field[subregion[0][0]:subregion[0][1]:sample[0],
#                                     h_slice,
#                                     subregion[1][0]:subregion[1][1]:sample[1], 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')
#
#     elif anatomical_plane == 'coronal':
#         ax0.quiver(id_field.field[h_slice,
#                                   subregion[0][0]:subregion[0][1]:sample[0],
#                                   subregion[1][0]:subregion[1][1]:sample[1], 0, 0],
#                    id_field.field[h_slice,
#                                   subregion[0][0]:subregion[0][1]:sample[0],
#                                   subregion[1][0]:subregion[1][1]:sample[1], 0, 1],
#                    input_field_copy.field[h_slice,
#                                           subregion[0][0]:subregion[0][1]:sample[0],
#                                           subregion[1][0]:subregion[1][1]:sample[1], 0, 0],
#                    input_field_copy.field[h_slice,
#                                           subregion[0][0]:subregion[0][1]:sample[0],
#                                           subregion[1][0]:subregion[1][1]:sample[1], 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')
#
#     else:
#         raise TypeError('Anatomical_plane must be axial, sagittal or coronal')
#
#     if long_title:
#         ax0.set_title(title_input + ', ' + str(anatomical_plane) + ' plane, slice ' + str(h_slice))
#     else:
#         ax0.set_title(title_input)
#
#     fig.set_tight_layout(True)
#     return fig
#
#
# ###  Methods for jacobian 2d ###
#
#
# def see_jacobian_of_a_field_2d(input_jac,
#                                sample=(1, 1),
#                                window_title_input='quiver',
#                                title_input= '2d vector field',
#                                fig_tag=1, scale=1,
#                                subtract_id=False):
#     """
#     Fields visualizer.
#     :param sample: does not plot all the vertical lines but a subspace
#     :return: plot the quiver of the svf in the sub-region we are interested in.
#     """
#
#     if not len(input_jac.shape) == 5:
#         raise TypeError('Wrong input size for a field')
#
#     if not input_jac.dim == 2 and input_jac.shape[4] == 4:
#             raise TypeError('See field 2d works only for 2d to 4d fields (as 2d jacobian).')
#
#     # Creating new sliced fields (jacobian is ordered column major):
#     half_shape_jac_0 = list(input_jac.shape[:])
#     half_shape_jac_0[4] = 2
#     jac_f1 = Field.generate_id(shape=half_shape_jac_0)  # elem 0 and 2
#     jac_f2 = Field.generate_id(shape=half_shape_jac_0)  # elem 1 and 3
#
#     jac_f1.field[..., 0] = copy.deepcopy(input_jac.field[..., 0])
#     jac_f1.field[..., 1] = copy.deepcopy(input_jac.field[..., 2])
#
#     jac_f2.field[..., 0] = copy.deepcopy(input_jac.field[..., 1])
#     jac_f2.field[..., 1] = copy.deepcopy(input_jac.field[..., 3])
#
#     id_field = Field.generate_id_from_obj(jac_f1)
#
#     # copy to subtract the identity without modifying the original copy:
#     jac_f1_c = copy.deepcopy(jac_f1)
#     jac_f2_c = copy.deepcopy(jac_f2)
#
#     if subtract_id:
#         jac_f1_c -= id_field
#         jac_f2_c -= id_field
#
#     fig = plt.figure(fig_tag)
#     ax_f1 = fig.add_subplot(121)
#     ax_f2 = fig.add_subplot(122)
#     fig.canvas.set_window_title(window_title_input)
#
#     ax_f1.quiver(id_field.field[::sample[0], ::sample[1], 0, 0, 0],
#                      id_field.field[::sample[0], ::sample[1], 0, 0, 1],
#                      jac_f1_c.field[::sample[0], ::sample[1], 0, 0, 0],
#                      jac_f1_c.field[::sample[0], ::sample[1], 0, 0, 1],
#                      color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')
#     ax_f2.quiver(id_field.field[::sample[0], ::sample[1], 0, 0, 0],
#                      id_field.field[::sample[0], ::sample[1], 0, 0, 1],
#                      jac_f2_c.field[::sample[0], ::sample[1], 0, 0, 0],
#                      jac_f2_c.field[::sample[0], ::sample[1], 0, 0, 1],
#                      color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale, scale_units='xy')
#
#     ax_f1.set_title(title_input + ', df_1/dx vs df_1/dy')
#     ax_f2.set_title(title_input + ', df_2/dx vs df_2/dy')
#
#     fig.set_tight_layout(True)
#     return fig
#
#
# def see_2_jacobian_of_2_fields_2d(input_jac_0, input_jac_1,
#                                   anatomical_plane_0='axial', anatomical_plane_1='axial',
#                                   h_slice_0=0, h_slice_1=0,
#                                   sample_0=(1, 1), sample_1=(1, 1),
#                                   window_title_input='quiver 2 screens',
#                                   title_input_0= 'Vector field',title_input_1= 'Vector field',
#                                   long_title_0=False, long_title_1=False,
#                                   fig_tag=1, scale_0=1, scale_1=1,
#                                   subtract_id_0=False, subtract_id_1=False):
#     """
#     Visualiser for a couple of 2d jacobian vector fields (from 2d to 4d vector fields).
#     4d fields are divided into 2 2d fields for its visualisation, as done in see_jacobian_of_a_field_2d
#     for only one jacobian field.
#     """
#
#     # Test if the input are meaningful:
#     if not len(input_jac_0.shape) == len(input_jac_1.shape) == 5:
#         raise TypeError('Wrong input size for the given fields')
#
#     if not (input_jac_0.dim == input_jac_1.dim == 2 and input_jac_0.shape[4] == input_jac_0.shape[4] == 4):
#         raise TypeError('See field 2d works only for a couple of 2d to 4d fields (as 2d jacobian).')
#
#     # Creating new sliced fields (jacobian is ordered column major):
#     half_shape_jac_0 = list(input_jac_0.shape[:])
#     half_shape_jac_0[4] = 2
#     jac_0_f1 = Field.generate_id(shape=half_shape_jac_0)  # elem 0 and 2
#     jac_0_f2 = Field.generate_id(shape=half_shape_jac_0)  # elem 1 and 3
#
#     half_shape_jac_1 = list(input_jac_1.shape[:])
#     half_shape_jac_1[4] = 2
#     jac_1_f1 = Field.generate_id(shape=half_shape_jac_1)  # elem 0 and 2
#     jac_1_f2 = Field.generate_id(shape=half_shape_jac_1)  # elem 1 and 3
#
#     jac_0_f1.field[..., 0] = copy.deepcopy(input_jac_0.field[..., 0])
#     jac_0_f1.field[..., 1] = copy.deepcopy(input_jac_0.field[..., 2])
#
#     jac_0_f2.field[..., 0] = copy.deepcopy(input_jac_0.field[..., 1])
#     jac_0_f2.field[..., 1] = copy.deepcopy(input_jac_0.field[..., 3])
#
#     jac_1_f1.field[..., 0] = copy.deepcopy(input_jac_1.field[..., 0])
#     jac_1_f1.field[..., 1] = copy.deepcopy(input_jac_1.field[..., 2])
#
#     jac_1_f2.field[..., 0] = copy.deepcopy(input_jac_1.field[..., 1])
#     jac_1_f2.field[..., 1] = copy.deepcopy(input_jac_1.field[..., 3])
#
#     id_field_0 = Field.generate_id_from_obj(jac_0_f1)
#     id_field_1 = Field.generate_id_from_obj(jac_1_f1)
#
#     # copy to subtract the identity without modifying the original copy:
#     jac_0_f1_c = copy.deepcopy(jac_0_f1)
#     jac_0_f2_c = copy.deepcopy(jac_0_f2)
#     jac_1_f1_c = copy.deepcopy(jac_1_f1)
#     jac_1_f2_c = copy.deepcopy(jac_1_f2)
#
#     if subtract_id_0:
#         jac_0_f1_c -= id_field_0
#         jac_0_f2_c -= id_field_0
#
#     if subtract_id_1:
#         jac_1_f1_c -= id_field_1
#         jac_1_f2_c -= id_field_1
#
#     # Initialize figure:
#     fig = plt.figure(fig_tag)
#     ax0 = fig.add_subplot(221)
#     ax1 = fig.add_subplot(222)
#     ax2 = fig.add_subplot(223)
#     ax3 = fig.add_subplot(224)
#
#     fig.canvas.set_window_title(window_title_input)
#
#     # figure 0: jac 0 f1 and f2
#     if anatomical_plane_0 == 'axial':
#         ax0.quiver(id_field_0.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
#                    id_field_0.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
#                    jac_0_f1_c.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
#                    jac_0_f1_c.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
#         ax1.quiver(id_field_1.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
#                    id_field_1.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
#                    jac_0_f2_c.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
#                    jac_0_f2_c.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')
#
#     elif anatomical_plane_0 == 'sagittal':
#         ax0.quiver(id_field_0.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
#                    id_field_0.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
#                    jac_0_f1_c.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
#                    jac_0_f1_c.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
#         ax1.quiver(id_field_0.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 0],
#                    id_field_0.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 1],
#                    jac_0_f2_c.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 0],
#                    jac_0_f2_c.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')
#
#     elif anatomical_plane_0 == 'coronal':
#         ax0.quiver(id_field_0.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
#                    id_field_0.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
#                    jac_0_f1_c.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
#                    jac_0_f1_c.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
#         ax1.quiver(id_field_1.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
#                    id_field_1.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
#                    jac_0_f2_c.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
#                    jac_0_f2_c.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')
#     else:
#         raise TypeError('anatomical_plane_0 must be axial, sagittal or coronal')
#
#     if long_title_0:
#         ax0.set_title(title_input_0 + 'df_1/dx vs df_1/dy' + ', ' + str(anatomical_plane_0) + ' plane, slice ' + str(h_slice_0))
#         ax1.set_title(title_input_0 + 'df_2/dx vs df_2/dy' + ', ' + str(anatomical_plane_1) + ' plane, slice ' + str(h_slice_1))
#     else:
#         ax0.set_title(title_input_0 + 'df_1/dx vs df_1/dy')
#         ax1.set_title(title_input_0 + 'df_2/dx vs df_2/dy')
#
#     # figure 2: jac 1 f1 and f2
#     if anatomical_plane_0 == 'axial':
#         ax2.quiver(id_field_0.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
#                    id_field_0.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
#                    jac_1_f1_c.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 0],
#                    jac_1_f1_c.field[::sample_0[0], ::sample_0[1], h_slice_0, 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
#         ax3.quiver(id_field_1.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
#                    id_field_1.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
#                    jac_1_f2_c.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 0],
#                    jac_1_f2_c.field[::sample_1[0], ::sample_1[1], h_slice_1, 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')
#
#     elif anatomical_plane_0 == 'sagittal':
#         ax2.quiver(id_field_0.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
#                    id_field_0.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
#                    jac_1_f1_c.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 0],
#                    jac_1_f1_c.field[::sample_0[0], h_slice_0, ::sample_0[1], 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
#         ax3.quiver(id_field_0.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 0],
#                    id_field_0.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 1],
#                    jac_1_f2_c.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 0],
#                    jac_1_f2_c.field[::sample_1[0], h_slice_0, ::sample_1[1], 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')
#
#     elif anatomical_plane_0 == 'coronal':
#         ax2.quiver(id_field_0.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
#                    id_field_0.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
#                    jac_1_f1_c.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 0],
#                    jac_1_f1_c.field[h_slice_0, ::sample_0[0], ::sample_0[1], 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_0, scale_units='xy')
#         ax3.quiver(id_field_1.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
#                    id_field_1.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
#                    jac_1_f2_c.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 0],
#                    jac_1_f2_c.field[h_slice_1, ::sample_1[0], ::sample_1[1], 0, 1],
#                    color='b', linewidths=0.01, width=0.03, units='xy', angles='xy', scale=scale_1, scale_units='xy')
#     else:
#         raise TypeError('anatomical_plane_0 must be axial, sagittal or coronal')
#
#     if long_title_1:
#         ax2.set_title(title_input_1 + 'df_1/dx vs df_1/dy' + ', ' + str(anatomical_plane_0) + ' plane, slice ' + str(h_slice_0))
#         ax3.set_title(title_input_1 + 'df_2/dx vs df_2/dy' + ', ' + str(anatomical_plane_1) + ' plane, slice ' + str(h_slice_1))
#     else:
#         ax2.set_title(title_input_1 + 'df_1/dx vs df_1/dy')
#         ax3.set_title(title_input_1 + 'df_2/dx vs df_2/dy')
#
#     fig.set_tight_layout(True)
#     return fig
#
#

#
#
# def triptych_quiver_quiver_quiver(vector_field_1,
#                                   vector_field_2,
#                                   vector_field_3,
#                                   fig_tag=5,
#                                   input_fig_size=(15, 5),
#                                   window_title_input='triptych'):
#
#     fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
#     fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)
#
#     font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
#
#     fig.canvas.set_window_title(window_title_input)
#
#     ax_1 = plt.subplot(131)
#     x1, y1 = np.meshgrid(np.arange(vector_field_1.shape[0]), np.arange(vector_field_1.shape[1]))
#     ax_1.quiver(y1,
#                 x1,
#                 vector_field_1[:, :, 0, 0, 0],
#                 vector_field_1[:, :, 0, 0, 1], scale=1, scale_units='xy')
#     ax_1.axes.xaxis.set_ticklabels([])
#     ax_1.axes.yaxis.set_ticklabels([])
#     ax_1.set_xlabel('(a)', fontdict=font)
#     ax_1.set_aspect('equal')
#
#     ax_2 = plt.subplot(132)
#     x2, y2 = np.meshgrid(np.arange(vector_field_2.shape[0]), np.arange(vector_field_2.shape[1]))
#     ax_2.quiver(y2,
#                 x2,
#                 vector_field_2[:, :, 0, 0, 0],
#                 vector_field_2[:, :, 0, 0, 1], scale=1, scale_units='xy')
#     ax_2.axes.xaxis.set_ticklabels([])
#     ax_2.axes.yaxis.set_ticklabels([])
#     ax_2.set_xlabel('(b)', fontdict=font)
#     ax_2.set_aspect('equal')
#
#     ax_3 = plt.subplot(133)
#     x3, y3 = np.meshgrid(np.arange(vector_field_3.shape[0]), np.arange(vector_field_3.shape[2]))
#     ax_3.quiver(y3,
#                 x3,
#                 vector_field_3[:, 0, :, 0, 0],
#                 vector_field_3[:, 0, :, 0, 1], scale=1, scale_units='xy')
#     ax_3.axes.xaxis.set_ticklabels([])
#     ax_3.axes.yaxis.set_ticklabels([])
#     ax_3.set_xlabel('(c)', fontdict=font)
#     ax_3.set_aspect('equal')
#
#     return fig
#
#
# def quadrivium_quiver(vector_field_1,
#                       vector_field_2,
#                       vector_field_3,
#                       vector_field_4,
#                       fig_tag=5,
#                       input_fig_size=(18, 4),
#                       window_title_input='triptych'):
#
#     fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
#     fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)
#
#     font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
#
#     fig.canvas.set_window_title(window_title_input)
#
#     ax_1 = plt.subplot(141)
#     x1, y1 = np.meshgrid(np.arange(vector_field_1.shape[0]), np.arange(vector_field_1.shape[1]))
#     ax_1.quiver(y1,
#                 x1,
#                 vector_field_1[:, :, 0, 0, 0],
#                 vector_field_1[:, :, 0, 0, 1], scale=1, scale_units='xy')
#     ax_1.axes.xaxis.set_ticklabels([])
#     ax_1.axes.yaxis.set_ticklabels([])
#     ax_1.set_xlabel('(a)', fontdict=font)
#     ax_1.set_aspect('equal')
#
#     ax_2 = plt.subplot(142)
#     x2, y2 = np.meshgrid(np.arange(vector_field_2.shape[0]), np.arange(vector_field_2.shape[1]))
#     ax_2.quiver(y2,
#                 x2,
#                 vector_field_2[:, :, 0, 0, 0],
#                 vector_field_2[:, :, 0, 0, 1], scale=1, scale_units='xy')
#     ax_2.axes.xaxis.set_ticklabels([])
#     ax_2.axes.yaxis.set_ticklabels([])
#     ax_2.set_xlabel('(b)', fontdict=font)
#     ax_2.set_aspect('equal')
#
#     ax_3 = plt.subplot(143)
#     x3, y3 = np.meshgrid(np.arange(vector_field_3.shape[0]), np.arange(vector_field_3.shape[1]))
#     ax_3.quiver(y3,
#                 x3,
#                 vector_field_3[:, :, 0, 0, 0],
#                 vector_field_3[:, :, 0, 0, 1], scale=1, scale_units='xy')
#     ax_3.axes.xaxis.set_ticklabels([])
#     ax_3.axes.yaxis.set_ticklabels([])
#     ax_3.set_xlabel('(c)', fontdict=font)
#     ax_3.set_aspect('equal')
#
#     ax_4 = plt.subplot(144)
#     x4, y4 = np.meshgrid(np.arange(vector_field_4.shape[0]), np.arange(vector_field_4.shape[2]))
#
#     print x4.size, y4.size
#
#     ax_4.quiver(y4,
#                 x4,
#                 vector_field_4[:, 0, :, 0, 0],
#                 vector_field_4[:, 0, :, 0, 1], scale=1, scale_units='xy')
#     ax_4.axes.xaxis.set_ticklabels([])
#     ax_4.axes.yaxis.set_ticklabels([])
#     ax_4.set_xlabel('(d)', fontdict=font)
#     ax_4.set_aspect('equal')
#
#     return fig
#
#
# def triptych_quiver_quiver_image(vector_field_1,
#                                  vector_field_2,
#                                  image_1,
#                                  fig_tag=5,
#                                  input_fig_size=(15, 5),
#                                  window_title_input='triptych'):
#
#     fig = plt.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
#     fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)
#
#     font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
#
#     fig.canvas.set_window_title(window_title_input)
#
#     ax_1 = plt.subplot(131)
#     x1, y1 = np.meshgrid(np.arange(vector_field_1.shape[0]), np.arange(vector_field_1.shape[1]))
#     ax_1.quiver(y1, x1,
#                 vector_field_1[:, :, 0, 0, 0], vector_field_1[:, :, 0, 0, 1],
#                 scale=1, scale_units='xy')
#     ax_1.axes.xaxis.set_ticklabels([])
#     ax_1.axes.yaxis.set_ticklabels([])
#     ax_1.set_xlabel('(a)', fontdict=font)
#     ax_1.set_aspect('equal')
#
#     ax_2 = plt.subplot(132)
#     x2, y2 = np.meshgrid(np.arange(vector_field_2.shape[0]), np.arange(vector_field_2.shape[1]))
#     ax_2.quiver(y2,
#                 x2,
#                 vector_field_2[:, :, 0, 0, 0],
#                 vector_field_2[:, :, 0, 0, 1], scale=1, scale_units='xy')
#     ax_2.axes.xaxis.set_ticklabels([])
#     ax_2.axes.yaxis.set_ticklabels([])
#     ax_2.set_xlabel('(b)', fontdict=font)
#     ax_2.set_aspect('equal')
#
#     ax_3 = plt.subplot(133)
#     ax_3.imshow(image_1, cmap='Greys',  interpolation='nearest', origin='lower')
#     ax_3.axes.xaxis.set_ticklabels([])
#     ax_3.axes.yaxis.set_ticklabels([])
#     ax_3.set_xlabel('(c)', fontdict=font)
#     ax_3.set_aspect('equal')
#
#     return fig
