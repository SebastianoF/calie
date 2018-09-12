"""
Classes for projective real Lie group, general and special, of dimension d.
Default value d=2.

https://en.wikipedia.org/wiki/Projective_linear_group
"""
import numpy as np
from scipy.linalg import expm, logm

from VECtorsToolkit.tools.transformations.pgl2_a import Pgl2A


class Pgl2G(object):
    """
    Real projective general/special linear Lie group of dimension 2.
    Each element is a (d+1)x(d+1) real matrix with non-zero determinant defined up to an element of
    the scalar transformations given by c*I for c constant and I identity matrix.
    It is meant to be generated as exponential of an element of the class pgl_2 with its method exponentiate.
    """
    def __init__(self, d=2, m=np.eye(3), special=False):
        if not isinstance(m, np.ndarray):
            raise IOError
        if np.array_equal(m.shape, [d+1]*2) and not np.linalg.det(m) == 0:
            self.matrix = m
            self.dim = d
            self.special = special
        else:
            raise IOError

    def shape(self):
        return self.matrix.shape

    shape = property(shape)

    def centered_matrix(self, c):
        """
        :param c: center = np.array([x_c, y_c])
        Returns: the matrix correspondent to self, centered in c. Non destructive over c.
        """
        h = self.matrix[:]
        h_prime = np.zeros_like(self.matrix)
        d = self.dim
        if isinstance(c, list):
            c = np.array(c)
        # den = one - Bc
        den = 1 - h[d, :-1].dot(c)

        # A prime
        h_prime[:-1, :-1] = (h[:-1, :-1] + np.kron(c.reshape(1, 2), h[d, :-1].reshape(2, 1)))/den
        # B prime
        h_prime[d, :-1] = (h[d, :-1])/den
        # T prime
        h_prime[:-1, 2] = (-(h[:-1, :-1]).dot(c) - h[d, :-1].dot(c) * c + c)/den

        h_prime[d, d] = 1

        return h_prime

    def centered(self, c):
        """
        Non destructive, provide a new homography as self, translated on c.
        :param c:
        :return:
        """
        return Pgl2G(d=self.dim, m=self.centered_matrix(c))

    def centering(self, c):
        """
        destructive, center the given homography to the given center
        :param c:
        :return:
        """
        self.matrix = self.centered_matrix(c)


def randomgen(d=2, center=None, scale_factor=None, sigma=1.0, special=False):
    """
    H = [A, T; B, 1]
    :param d:
    :param center: if we want to center the given matrix.
    :param scale_factor:
    :param sigma:
    :param special:
    :return:
    """
    random_h = sigma*np.random.randn(d+1,  d+1)
    # select one equivalence class.
    random_h[-1, -1] = 1
    # Ensure its matrix logarithm will have real entries.
    random_h = expm(random_h)

    if scale_factor is not None:
        random_h = scale_factor * random_h

    if center is not None:
        random_h = Pgl2G(d=d, m=random_h).centered_matrix(center)

    if special:
        random_h[0, 0] = -1 * np.sum(np.diagonal(random_h)[1:])

    random_h[-1, -1] = 1.

    return Pgl2G(d=d, m=random_h, special=special)


def pgl2a_log(pgl2a):
    return Pgl2A(pgl2a.dim, logm(pgl2a.matrix)[:], special=pgl2a.special)


def randomgen_special(d=2, center=None, scale_factor=None, sigma=1.0, special=False,
                      get_as_matrix=False):
    """
    :param d: dimension of the homography in pgl by default or in psl
    :param center: center of the homography, if any
    :param scale_factor: scale factor of the homography
    :param sigma: sigma for the random values of the initial matrix.
    :param special: if the homography is in psl (True) or in pgl (False, default)
    :param get_as_matrix: if true output is a matrix.
    :return: [h_g, h_a]random homography (in the GROUP) and the corresponding in the algebra h_g = expm(h_a)
    """
    if special:
        h_g = randomgen(d=d, center=center, scale_factor=scale_factor, sigma=sigma)
        h_a = pgl2a_log(h_g)

    else:
        h_g = randomgen(d=d, center=center, scale_factor=scale_factor, sigma=sigma)
        h_a = pgl2a_log(h_g)

    if get_as_matrix:
        return h_g.matrix, h_a.matrix
    else:
        return h_g, h_a

'''

# methods to get above elements as matrices: it masks the classes!
def get_random_hom_g(d=2, center=None, scale_factor=None, sigma=1.0, special=False):
    """
    :param d: dimension of the homography in pgl by default or in psl
    :param center: center of the homography, if any
    :param scale_factor: scale factor of the homography
    :param sigma: sigma for the random values of the initial matrix.
    :param special: if the homography is in psl (True) or in pgl (False, default)
    :return: [h_g, h_a]random homography (in the GROUP) and the corresponding in the algebra h_g = expm(h_a)
    """
    if special:
        h_g = ProjectiveGroup.randomgen(d=d, center=center, scale_factor=scale_factor, sigma=sigma)
        h_a = h_g.logaritmicate()

    else:
        h_g = ProjectiveGroup.randomgen(d=d, center=center, scale_factor=scale_factor, sigma=sigma)
        h_a = h_g.logaritmicate()

    return h_g, h_a


def get_random_hom_a_matrices(d=2, scale_factor=None, sigma=1.0, special=False):
    """
    :param d: dimension of the homography in pgl by default or in psl
    :param center: center of the homography, if any
    :param scale_factor: scale factor of the homography
    :param sigma: sigma for the random values of the initial matrix.
    :param special: if the homography is in psl (True) or in pgl (False, default)
    :return: [h_g, h_a]random homography (in the GROUP) and the corresponding in the algebra h_g = expm(h_a)
    """

    h_a = ProjectiveAlgebra.randomgen(d=d, scale_factor=scale_factor, sigma=sigma, special=special)
    h_g = h_a.exponentiate()

    return h_a.matrix, h_g.matrix



def get_random_hom_matrices(d=2, center=None, random_kind='diag', scale_factor=None, sigma=1.0, special=False):
    """
    :param d: dimension of the homography in pgl by default or in psl
    :param center: center of the homography, if any
    :param random_kind
    :param scale_factor: scale factor of the homography
    :param sigma: sigma for the random values of the initial matrix.
    :param special: if the homography is in psl (True) or in pgl (False, default)
    :return: [h_g, h_a]random homography (in the GROUP) and the corresponding in the algebra h_g = expm(h_a)
    AS MATRICES!
    """
    if special:
        h_g = ProjectiveGroup.randomgen(d=d, center=center, scale_factor=scale_factor, sigma=sigma)
        h_a = h_g.logaritmicate()

    else:
        h_g = ProjectiveGroup.randomgen(d=d, center=center, scale_factor=scale_factor, sigma=sigma)
        h_a = h_g.logaritmicate()

    return h_g.matrix, h_a.matrix

'''