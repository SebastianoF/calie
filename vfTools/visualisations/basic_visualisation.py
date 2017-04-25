import copy
import matplotlib.pyplot as plt

from src.tools.auxiliary.sanity_checks import check_is_vector_field
from src.tools.auxiliary.generators_vector_fields import generate_identity_lagrangian

from matplotlib.widgets import Slider, Button, RadioButtons


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
              h_slice=0, sample=(1, 1),
              window_title_input='quiver',
              title_input= '2d vector field',
              long_title=False,
              fig_tag=1, scale=1,
              subtract_id=False,
              input_color='b',
              annotate=None, annotate_position=(1, 1)):

    check_is_vector_field(input_vf)
    d = input_vf.shape[-1]
    if not d == 2:
            raise TypeError('See field 2d works only for 2d to 2d fields.')

    id_field = generate_identity_lagrangian(list(input_vf.shape[:d]))

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


