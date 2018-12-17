import matplotlib.pyplot as plt
import numpy as np

from VECtorsToolkit.visualisations.fields import fields_at_the_window



def f_vcon(t, x):
    """
    real eigenvalue both positive or both negative: stable node.
    """
    t = float(t)
    x = [float(z) for z in x]
    sigma = 0.2
    I = 0.99
    w = 0.9
    alpha = 0.04
    tx, ty = -1, -1

    return alpha * (x[1]) + tx, alpha * (-1 * sigma * x[1] + I + w * np.cos(x[0])) + ty


def vf_linear(t, x, m, alpha=1.):
    """
    Linear 2d ODE from the matrix
    [a b]           [tx]
    [c d] * alpha + [ty]
    :param t: time parameter
    :param x: [x,y] 2d array
    :param m: numpy array with rotation + translation in homogeneous coordinates:
    [a b tx]
    [c d ty]
    [0 0  1]
    :param alpha: scaling factor for the rotational part
    :return:
    """
    t = float(t)
    x = [float(z) for z in x]
    return alpha * (m[0, 0]*x[0] + m[0, 1]*x[1] + m[0, 2]), alpha * (m[1, 0]*x[0] + m[1, 1]*x[1] + m[1, 2])


if __name__ == '__main__':

    see_vector_fields = {'vcon'             : True,
                         'descartes_folium' : True}

    shape = (20, 20, 1, 1, 2)

    if see_vector_fields['vcon']:
        field_vcon = np.zeros(shape)

        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                field_vcon[i, j, 0, 0, :] = f_vcon(1, [i, j])

        fields_at_the_window.see_field(field_vcon, scale=1, input_color='r', fig_tag=2, annotate='', annotate_position=[-4, 1])

    if see_vector_fields['descartes_folium']:
        field_descartes = np.zeros(shape)
        m_descartes = np.array([[2.5, -2, 5], [2, -2, 5], [0, 0, 1]])
        alpha_descartes = 0.2

        for i in range(0, 20):
            for j in range(0, 20):
                field_descartes[i, j, 0, 0, :] = vf_linear(1, [i, j], m_descartes, alpha=alpha_descartes)

        fields_at_the_window.see_field(field_descartes, scale=1, input_color='r', fig_tag=1, annotate='', annotate_position=[-4, 1])

    plt.show()
