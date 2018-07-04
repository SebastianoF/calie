import numpy as np



def check_omega(omega):
    """
    Sanity check for omega.
    """
    d = len(omega)
    if not all(isinstance(x, int) for x in omega):
        raise IOError('Input omega of the wrong type. \n' + omega)
    if not (d == 2 or d == 3):
        raise IOError('Input omega of the wrong dimension. \n' + omega)
    return d


def check_is_vector_field(input_obj):

    is_vector_field = True

    # check if is a numpy.ndarray
    if not isinstance(input_obj, np.ndarray):
        is_vector_field  = False

    # check shape compatibility with the accepted vector fields structures
    if not len(input_obj.shape) == 5:
        is_vector_field  = False

    # check if the dimension of omega (domain) is a multiple of the dimension of the codomain
    if input_obj.shape[2] == 1: d = 2
    else: d = 3

    if not input_obj.shape[-1] % d == 0:
        is_vector_field  = False

    if not is_vector_field:
        raise IOError('Input numpy array is not a vector field.')
    else:
        return d


def get_v_shape_from_omega(omega, t=0):

    d = check_omega(omega)
    v_shape = list(omega) + [1] * (3 - d) + [t, d]
    return v_shape


def get_omega_from_vf(input_vf):

    if check_is_vector_field(input_vf):

        vf_shape = input_vf.shape
        if vf_shape[2] == 1:
            omega = list(vf_shape[:2])
        else:
            omega = list(vf_shape[:3])
        return omega
