"""
Classes for projective real Lie algebra, general and special, of dimension d.
Default value d=2.

https://en.wikipedia.org/wiki/Projective_linear_group
"""
import numpy as np
from scipy.linalg import expm

from VECtorsToolkit.tools.transformations.pgl2_g import Pgl2G


class Pgl2A(object):
    """
    Real projective general/special linear Lie algebra of dimension d.
    Each element is a (d+1)x(d+1) real matrix defined up to a constant.
    Its exponential is given through the numerical approximation expm.
    If special trace must be zero.
    """
    def __init__(self, d=2, m=np.zeros([3, 3]), special=False):
        """
        """
        if np.array_equal(m.shape, [d + 1] * 2):
            self.matrix = m  # - np.min(m)*np.ones([d+1, d+1])
            self.dim = d
            self.special = special
        else:
            raise IOError
        if special is True and not np.abs(np.trace(m)) < 1e-4:
            raise IOError('input matrix is not in the special projective group')

    def shape(self):
        return self.matrix.shape

    shape = property(shape)

    def ode_solution(self, init_cond=np.array([0, 0, 1]), affine_coordinates=True):
        s = expm(self.matrix).dot(init_cond)
        if affine_coordinates:
            return s[0:self.dim]/s[self.dim]  # the projective coordinate is the last one
        else:
            return s


def randomgen(d=2, scale_factor=None, sigma=1.0, special=False):
    """
    Generate a random element in the projective linear algebra
    :param d:
    :param scale_factor:
    :param sigma:
    :param special:
    :return:
    """
    random_h = sigma*np.random.randn(d+1,  d+1)

    if scale_factor is not None:
        random_h = scale_factor * random_h

    if special:
        random_h[0, 0] = -1 * np.sum(np.diagonal(random_h)[1:])

    return Pgl2A(d=d, m=random_h, special=special)


def pgl2_a_exp(pgl2a):
    return Pgl2G(pgl2a.dim, expm(pgl2a.matrix), special=pgl2a.special)


'''
 if random_kind == 'diag':
                    random_h[0, 0] = 1 - (random_h[0, 1] * y_c + random_h[0, 2] * z_c)/float(x_c)
                    random_h[1, 1] = 1 - (random_h[1, 0] * x_c + random_h[1, 2] * z_c)/float(y_c)
                    random_h[2, 2] = 1 - (random_h[2, 0] * x_c + random_h[2, 1] * y_c)/float(z_c)
                elif random_kind == 'skew':
                    random_h[0, 2] = ((1 - random_h[0, 0]) * x_c - random_h[0, 1] * y_c) / float(z_c)
                    random_h[1, 0] = ((1 - random_h[1, 1]) * y_c - random_h[1, 2] * z_c) / float(x_c)
                    random_h[2, 1] = ((1 - random_h[2, 2]) * z_c - random_h[2, 0] * x_c) / float(y_c)
                elif random_kind == 'transl':
                    random_h[0, 2] = (1 - random_h[0, 0]) * x_c - random_h[0, 1] * y_c
                    random_h[1, 2] = - random_h[1, 0] * x_c + (1 - random_h[1, 1]) * y_c
                    random_h[2, 2] = 1  # - random_h[2, 0] * x_c - random_h[2, 1] * y_c + 1

                elif random_kind == 'shift':

                    two_minus_Bc      = 2 - random_h[2, :-1].dot(np.array([x_c, y_c]).T)
                    T_plus_c_minus_Ac = random_h[:-1, 2] + np.array([x_c, y_c]).T - \
                          random_h[-1, :-1].dot(np.array([x_c, y_c]).T)

                    # A prime
                    random_h[:-1, :-1] = random_h[:-1, :-1]/float(two_minus_Bc)
                    # B prime
                    random_h[2, :-1] = random_h[2, :-1]/float(two_minus_Bc)
                    # T prime
                    random_h[:-1, 2] = T_plus_c_minus_Ac

                    random_h[2, 2] = 1

                elif random_kind == 's':


                    one_minus_Bc = 1 - random_h[2, :-1].dot(np.array([x_c, y_c]).T)

                    # A prime
                    random_h[:-1, :-1] = random_h[:-1, :-1]/float(two_minus_Bc)
                    # B prime
                    random_h[2, :-1] = random_h[2, :-1]/float(two_minus_Bc)
                    # T prime
                    random_h[:-1, 2] = T_plus_c_minus_Ac

                    random_h[2, 2] = 1

                    pass

                else:
                    raise IOError('kind not recognized')

                    if random_kind == 'diag':
                    random_h[0, 0] = 1 - (random_h[0, 1] * y_c + random_h[0, 2] * z_c + random_h[0, 3] * w_c)/float(x_c)
                    random_h[1, 1] = 1 - (random_h[1, 0] * x_c + random_h[1, 2] * z_c + random_h[1, 3] * w_c)/float(y_c)
                    random_h[2, 2] = 1 - (random_h[2, 0] * x_c + random_h[2, 1] * y_c + random_h[2, 3] * w_c)/float(z_c)
                    random_h[3, 3] = 1 - (random_h[3, 0] * x_c + random_h[3, 1] * y_c + random_h[3, 2] * z_c)/float(w_c)
                elif random_kind == 'skew':
                    random_h[0, 3] = ((1 - random_h[0, 0]) * x_c - random_h[0, 1] * y_c - random_h[0, 2]*z_c)/float(w_c)
                    random_h[1, 0] = ((1 - random_h[1, 1]) * y_c - random_h[1, 2] * z_c - random_h[1, 3]*w_c)/float(x_c)
                    random_h[2, 1] = ((1 - random_h[2, 2]) * z_c - random_h[2, 3] * w_c - random_h[2, 0]*x_c)/float(y_c)
                    random_h[3, 2] = ((1 - random_h[3, 3]) * w_c - random_h[3, 0] * x_c - random_h[3, 1]*y_c)/float(z_c)
                elif random_kind == 'transl':
                    random_h[0, 3] = (1 - random_h[0, 0]) * x_c - random_h[0, 1] * y_c - random_h[0, 2] * z_c
                    random_h[1, 3] = - random_h[1, 0] * x_c + (1 - random_h[1, 1]) * y_c - random_h[1, 2] * z_c
                    random_h[2, 3] = - random_h[2, 0] * x_c - random_h[2, 1] * y_c + (1 - random_h[2, 2]) * z_c
                    random_h[3, 3] = 1 #  - random_h[3, 0] * x_c - random_h[3, 1] * y_c - random_h[3, 2] * z_c  + 1
                elif random_kind == 'shift':

                    two_minus_Bc      = 2 - random_h[3, :-1].dot(np.array([x_c, y_c, z_c]).T)
                    T_plus_c_minus_Ac = random_h[:-1, 3] + np.array([x_c, y_c]).T - \
                                        random_h[-1, :-1].dot(np.array([x_c, y_c, z_c]).T)
 
                    # A prime
                    random_h[:-1, :-1] = random_h[:-1, :-1]/float(two_minus_Bc)
                    # B prime
                    random_h[3, :-1] = random_h[3, :-1]/float(two_minus_Bc)
                    # T prime
                    random_h[:-1, 3] = T_plus_c_minus_Ac

                    random_h[3, 3] = 1

                else:
                    raise IOError('kind not recognized')
'''