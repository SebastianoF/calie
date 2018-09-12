import numpy as np


def grid_generator(x_size=101,
                   y_size=101,
                   x_step=10,
                   y_step=10,
                   line_thickness=1):
    """
    Creates a grid image (3d numpy array) after specified input.
    :param x_size:
    :param y_size:
    :param x_step:
    :param y_step:
    :param line_thickness:
    :return:
    """
    m = np.zeros([x_size, y_size])
    # initial slow version. Refactor after testing
    for x in range(x_size):
        for y in range(y_size):
            if 0 <= x % x_step < line_thickness or 0 <= y % y_step < line_thickness:
                m[x, y] = 1

    return m
