import copy
from matplotlib import pyplot

from VECtorsToolkit.fields import queries as qr
from VECtorsToolkit.fields import generate_identities as gen_id


def triptych_image_quiver_image(image_1,
                                input_vf,
                                image_2,
                                fig_tag=5,
                                input_fig_size=(15, 5),
                                window_title_input='triptych',
                                h_slice=1,
                                sampling_svf=(1, 1),
                                anatomical_plane='axial',
                                subtract_id=False,
                                scale=1,
                                input_color='r',
                                line_arrowwidths=0.01,
                                arrowwidths=0.3,
                                trace_integral_curves=None):

    fig = pyplot.figure(fig_tag, figsize=input_fig_size, dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)

    font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}

    fig.canvas.set_window_title(window_title_input)

    # First axis on the left
    ax_1 = pyplot.subplot(131)
    ax_1.imshow(image_1, cmap='Greys', interpolation='nearest', origin='lower')
    # ax_1.axes.xaxis.set_ticklabels([])
    # ax_1.axes.yaxis.set_ticklabels([])
    ax_1.set_xlabel('(grid)', fontdict=font)
    ax_1.set_aspect('equal')

    # Central vector field
    ax_2 = pyplot.subplot(132)
    qr.check_is_vf(input_vf)
    id_field = gen_id.id_eulerian_like(input_vf)
    input_field_copy = copy.deepcopy(input_vf)
    if subtract_id:
        input_field_copy -= id_field

    if anatomical_plane == 'axial':
        ax_2.quiver(id_field[::sampling_svf[0], ::sampling_svf[1], h_slice, 0, 0],
                   id_field[::sampling_svf[0], ::sampling_svf[1], h_slice, 0, 1],
                   input_field_copy[::sampling_svf[0], ::sampling_svf[1], h_slice, 0, 0],
                   input_field_copy[::sampling_svf[0], ::sampling_svf[1], h_slice, 0, 1],
                   color=input_color, linewidths=line_arrowwidths, width=arrowwidths,
                   scale=scale, scale_units='xy', units='xy',
                   angles='xy')
        # ax_2.set_xlabel('x')
        # ax_2.set_ylabel('y')

    elif anatomical_plane == 'sagittal':
        ax_2.quiver(id_field[::sampling_svf[0], h_slice, ::sampling_svf[1], 0, 0],
                   id_field[::sampling_svf[0], h_slice, ::sampling_svf[1], 0, 1],
                   input_field_copy[::sampling_svf[0], h_slice, ::sampling_svf[1], 0, 0],
                   input_field_copy[::sampling_svf[0], h_slice, ::sampling_svf[1], 0, 1],
                   color=input_color, linewidths=line_arrowwidths, width=arrowwidths,
                   units='xy', angles='xy', scale=scale,
                   scale_units='xy')

    elif anatomical_plane == 'coronal':
        ax_2.quiver(id_field[h_slice, ::sampling_svf[0], ::sampling_svf[1], 0, 0],
                   id_field[h_slice, ::sampling_svf[0], ::sampling_svf[1], 0, 1],
                   input_field_copy[h_slice, ::sampling_svf[0], ::sampling_svf[1], 0, 0],
                   input_field_copy[h_slice, ::sampling_svf[0], ::sampling_svf[1], 0, 1],
                   color=input_color, linewidths=line_arrowwidths, width=arrowwidths,
                    units='xy', angles='xy', scale=scale,
                   scale_units='xy')
    else:
        raise TypeError('Anatomical_plane must be axial, sagittal or coronal')

    # Integral curves of the vector fields:
    if trace_integral_curves:
        pass


    # ax_2.axes.xaxis.set_ticklabels([])
    # ax_2.axes.yaxis.set_ticklabels([])
    ax_2.set_xlabel('(transformation)', fontdict=font)
    ax_2.set_aspect('equal')

    # Third axis:
    ax_3 = pyplot.subplot(133)
    ax_3.imshow(image_2, cmap='Greys',  interpolation='none', origin='lower')
    # ax_3.axes.xaxis.set_ticklabels([])
    # ax_3.axes.yaxis.set_ticklabels([])
    ax_3.set_xlabel('(transformed grid)', fontdict=font)
    ax_3.set_aspect('equal')

    fig.set_tight_layout(True)

    return fig
