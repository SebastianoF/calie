import numpy as np


def check_omega(omega):
    """
    Sanity check for omega.
    """
    d = len(omega)
    if not all(isinstance(x, int) for x in omega):
        raise IOError('Input omega {} of the wrong type. \n'.format(omega))
    if not (d == 2 or d == 3):
        raise IOError('Input omega  {} of the wrong dimension. \n'.format(omega))
    return d


def check_is_vf(input_obj):

    is_vector_field = True

    # check if is a numpy.ndarray
    if not isinstance(input_obj, np.ndarray):
        is_vector_field  = False

    # check shape compatibility with the accepted vector fields structures
    if not len(input_obj.shape) == 5:
        is_vector_field  = False

    # check if the dimension of omega (domain) is a multiple of the dimension of the codomain
    if input_obj.shape[2] == 1:
        d = 2
    else:
        d = 3

    if not input_obj.shape[-1] % d == 0:
        is_vector_field  = False

    if not is_vector_field:
        raise IOError('Input numpy array is not a vector field.')
    else:
        return d


def get_omega_from_vf(input_vf):

    if check_is_vf(input_vf):

        vf_shape = input_vf.shape
        if vf_shape[2] == 1:
            omega = list(vf_shape[:2])
        else:
            omega = list(vf_shape[:3])
        return omega


def vf_shape_from_omega_and_timepoints(omega, t=0):

    d = check_omega(omega)
    v_shape = list(omega) + [1] * (3 - d) + [t, d]
    return v_shape


def get_omega(input_vf):

    if check_is_vf(input_vf):

        vf_shape = input_vf.shape
        if vf_shape[2] == 1:
            omega = list(vf_shape[:2])
        else:
            omega = list(vf_shape[:3])
        return omega


def vf_norm(input_vf, passe_partout_size=1, normalized=False):
    """
    Returns the L2-norm of the discretised vector field.
    The computation makes sense only with svf.
    Based on the norm function from numpy.linalg of ord=2 for the vectorized matrix.
    The result can be computed with a passe partout
    (the discrete domain is reduced on each side by the same value, keeping the proportion
    of the original image) and can be normalized with the size of the domain.

    -> F vector field from a compact \Omega to R^d
    \norm{F} = (\frac{1}{|\Omega|}\int_{\Omega}|F(x)|^{2}dx)^{1/2}
    Discretisation:
    \Delta\norm{F} = \frac{1}{\sqrt{dim(x)dim(y)dim(z)}}\sum_{v \in \Delta\Omega}|v|^{2})^{1/2}
                   = \frac{1}{\sqrt{XYZ}}\sum_{i,j,k}^{ X,Y,Z}|a_{i,j,k}|^{2})^{1/2}

    -> f scalar field from \Omega to R, f is an element of the L^s space
    \norm{f} = (\frac{1}{|\Omega|}\int_{\Omega}f(x)^{2}dx)^{1/2}
    Discretisation:
    \Delta\norm{F} = \frac{1}{\sqrt{XYZ}}\sum_{i,j,k}^{ X,Y,Z} a_{i,j,k}^{2})^{1/2}

    Parameters:
    ------------
    :param input_vf: input vector field.
    :param passe_partout_size: size of the passe partout (rectangular mask, with constant offset on each side).
    :param normalized: if the result is divided by the normalization constant.
    """
    d = check_is_vf(input_vf)
    if passe_partout_size > 0:
        if d == 2:
            masked_field = input_vf[passe_partout_size:-passe_partout_size,
                                    passe_partout_size:-passe_partout_size, ...]
        else:
            masked_field = input_vf[passe_partout_size:-passe_partout_size,
                                    passe_partout_size:-passe_partout_size,
                                    passe_partout_size:-passe_partout_size, ...]
    else:
        masked_field = input_vf

    if normalized:
        # volume of the field after masking (to compute the normalization factor):
        mask_vol = (np.array(input_vf.shape[0:d]) - np.array([2 * passe_partout_size] * d)).clip(min=1)

        return np.linalg.norm(masked_field.ravel(), ord=2) / np.sqrt(np.prod(mask_vol))
    else:
        return np.linalg.norm(masked_field.ravel(), ord=2)
