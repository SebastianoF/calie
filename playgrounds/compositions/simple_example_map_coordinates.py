"""
Simple toy example to see how map_coordinates works.
"""
import numpy as np
from scipy.ndimage import interpolation as interp

if __name__ == '__main__':
    in_data = np.array([[0., -1., -2.],
                        [2., 1., 0.],
                        [4., 3., 2.]])  # z = 2.*x - 1.*y

    # want the second argument as a column vector (or a transposed row)
    # see on some points of the grid:
    print('At the point 0, 0 of the grid the function z is: ')
    print(interp.map_coordinates(in_data, np.array([[0., 0.]]).T, order=1))
    print('\nAt the point 0, 1 of the grid the function z is: ')
    print(interp.map_coordinates(in_data, np.array([[0., 1.]]).T, order=1))
    print('\nAt the point 0, 2 of the grid the function z is: ')
    print(interp.map_coordinates(in_data, np.array([[0., 2.]]).T, order=1))

    # see some points outside the grid
    print('\nAt the point 0.2, 0.2 of the grid, with linear interpolation z is:')
    print(interp.map_coordinates(in_data, np.array([[.2, .2]]).T, order=1))
    print('and it coincides with .2*.2 - .2')
    print('\nAt the point 0.2, 0.2 of the grid, with cubic interpolation z is:')
    print(interp.map_coordinates(in_data, np.array([[0.2, .2]]).T, order=3))
