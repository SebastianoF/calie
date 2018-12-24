import os
import nibabel as nib
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
    """
    Core method to check if the input object satisfy the definition of a vector field.
    This method allow to avoid defining a class for the vf object, and to keep the code simpler.
    :param input_obj:
    :return: Raise error if input does not satisfy the definition of vector field.
    """
    # check if is a numpy.ndarray
    if not isinstance(input_obj, np.ndarray):
        raise IOError('Input numpy array is not a vector field.')
    # check shape compatibility with the accepted vector fields structures
    if not len(input_obj.shape) == 5:
        raise IOError('Input numpy array is not a vector field, as it is not 5-dimensional')
    # store the image domain dimension
    if input_obj.shape[2] == 1:
        d = 2
    else:
        d = 3
    # Check that the last dimension is a multiple of the dimension of omega
    if not input_obj.shape[-1] % d == 0:
        raise IOError('Input numpy array is not a vector field')
    else:
        return d


def get_omega(input_vf):

    if check_is_vf(input_vf):

        vf_shape = input_vf.shape
        if vf_shape[2] == 1:
            omega = list(vf_shape[:2])
        else:
            omega = list(vf_shape[:3])
        return omega


def shape_from_omega_and_timepoints(omega, t=0):

    d = check_omega(omega)
    v_shape = list(omega) + [1] * (3 - d) + [t, d]
    return v_shape


def norm(input_vf, passe_partout_size=1, normalized=False):
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

    if d == 2:
        num = np.sum(np.sqrt(masked_field[..., 0] ** 2 + masked_field[..., 1] ** 2))
    else:
        num = np.sum(np.sqrt(masked_field[..., 0] ** 2 + masked_field[..., 1] ** 2 + masked_field[..., 2] ** 2))

    if normalized:
        den = np.prod(masked_field.shape[:3])
    else:
        den = 1
    return num / float(den)


def nib_to_omega(input_nib_image):
    """
    :param input_nib_image: nibabel image or path to a nifti image.
    :return: omega with the input image
    """
    if isinstance(input_nib_image, str):
        if not os.path.exists(input_nib_image):
            raise IOError('Input path {} does not exist.'.format(input_nib_image))
        im = nib.load(input_nib_image)
        omega = im.shape
    else:
        omega = input_nib_image.shape
    return omega
