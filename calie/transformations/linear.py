import numpy as np

from calie.transformations import se2


def get_taste(dm):
    """
    Get the classification of a matrix defining a tangent vector field of the
    form:
            | R | t |
            | - - - |
            | 0 | 0 |
    :param dm:  input tangent matrix
    :return: number from 1 to 6 corresponding to taste. see randomgen_linear_by_taste.
    """
    rot = dm[:2, :2]
    v, w = np.linalg.eig(rot)

    if v[0].imag < np.spacing(0) and v[1].imag < np.spacing(0):
        # Eigenvalues both real:
        l1 = v[0].real
        l2 = v[1].real

        if l1 > 0 and l2 > 0:  # Taste 1
            return 1
        elif l1 < 0 and l2 < 0:  # Taste 2
            return 2
        else:  # Taste 3
            return 3
    else:
        # Complex conjugate eigenvalues
        if v[0].real > np.spacing(0):  # Taste 4
            return 4
        elif v[0].real < np.spacing(0):  # Taste 5
            return 5
        else:  # Taste 6 - never get there in practice.
            return 6


def randomgen_linear_by_taste(sigma, taste, center=(0, 0)):
    """
    To create a linear (2D) stationary tangent vector field according to the proposed taste
    classification:
    + Taste 1 : real eigenvalues with positive signs. UNSTABLE NODE
    + Taste 2 : real eigenvalues with negative signs. STABLE NODE
    + Taste 3 : real eigenvalues with opposite signs. SADDLE
    + Taste 4 : complex conjugate with positive real part. OUTWARD SPIRAL
    + Taste 5 : complex conjugate with negative real part. INWARD SPIRAL
    + Taste 6 : complex conjugate with 0 real parts. CIRCLES, Based on se2.Se2G
    :param sigma: how further away going from identity.
    :param taste: int between 1 and 6 as in the classification.
    :param center: (float, float) center of the transformation.
    :return: 3x3 tangent space linear transformation matrix of the shape
            | R | t |
            | - - - |
            | 0 | 0 |
    """
    if taste not in range(1, 7):
        raise IOError('taste must be an integer number between 1 and 6.')

    if taste == 6:
        x_c, y_c = center
        theta = sigma * np.random.randn() + 2 * sigma

        tx = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
        ty = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

        m0 = se2.Se2G(theta, tx, ty)
        dm0 = se2.se2g_log(m0)
        return dm0.get_matrix

    else:
        found = False
        rot = np.zeros([2, 2])
        while not found:
            rot = np.random.randn(2, 2)
            if get_taste(rot - np.eye(2)) == taste:
                found = True

        dm = np.zeros([3, 3])
        dm[:2, :2] = (1 / sigma) * rot

        tx = (1 - dm[0, 0]) * center[0] - dm[0, 1] * center[1]
        ty = - dm[1, 0] * center[0] + (1 - dm[1, 1]) * center[1]

        dm[:2, 2] = np.array([tx, ty])

        dm = dm - np.eye(3)
        dm[2, 2] = 0

        return dm


def randomgen_linear(sigma, center=(0, 0)):
    """
    Get a random number between 1 and 6 and returns the matrix with the corresponding taste.
    :param sigma: how further away going from identity
    :param center: (float, float) center of the transformation.
    :return: 3x3 tangent space linear transformation matrix of the shape
            | R | t |
            | - - - |
            | 0 | 0 |
    """
    taste = np.random.choice([1, 2, 3, 4, 5, 6])
    return randomgen_linear_by_taste(sigma, taste, center=center)
