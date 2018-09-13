from math import sin, cos, sqrt
from random import uniform
import numpy as np

from VECtorsToolkit.tools.auxiliary.angles import mod_pipi


class Se2A(object):
    """
    Class for algebra se2 quotient over equivalent relation given by exp.
    To respect camel case convention and avoid confusion class for se(2) is called Se2A.
    It is the quotient algebra in which exp is well defined.
    Quotient projection is applied to each new instance by default, to have exp map bijective.
    """
    def __init__(self, theta, tx, ty):

        self.rotation_angle = mod_pipi(theta)

        if self.rotation_angle == theta:
            self.tx = tx
            self.ty = ty
        else:
            modfact = self.rotation_angle / theta
            self.tx = modfact * tx
            self.ty = modfact * ty

    def __quotient_projection__(self):
        """
        Maps the elements of se2_a in the quotient over the
        equivalence relation defined by exp, in order to have
        exp well defined.
        a ~ b iff exp(a) == exp(b)
        Here se2_a is intended as the quotient se2_a over the
        above equivalence relation.
        :param self: element of se2_a
        """
        theta_quot = mod_pipi(self.rotation_angle)
        if self.rotation_angle == theta_quot:
            tx_quot = self.tx
            ty_quot = self.ty
        else:
            modfact = mod_pipi(self.rotation_angle) / self.rotation_angle
            tx_quot = self.tx * modfact
            ty_quot = self.ty * modfact
        return Se2A(theta_quot, tx_quot, ty_quot)

    quot = property(__quotient_projection__)

    def __get_matrix__(self):
        # using the quotient data to create the matrix.
        theta = self.quot.rotation_angle
        tx = self.quot.tx
        ty = self.quot.ty

        a1 = [0, -theta, tx]
        a2 = [theta, 0, ty]
        a3 = [0, 0, 0]
        return np.array([a1, a2, a3])

    get_matrix = property(__get_matrix__)

    def __get_restricted__(self):
        # get rotation and translation in a list from quotient.
        theta = self.quot.rotation_angle
        tx = self.quot.tx
        ty = self.quot.ty
        return [theta, tx, ty]

    get = property(__get_restricted__)

    def __add__(self, element2):
        alpha1 = self.rotation_angle
        x1 = self.tx
        y1 = self.ty
        alpha2 = element2.rotation_angle
        x2 = element2.tx
        y2 = element2.ty
        alpha_sum = alpha1+alpha2
        return Se2A(alpha_sum, x1 + x2, y1 + y2).quot

    def __sub__(self, element2):
        alpha1 = self.rotation_angle
        x1 = self.tx
        y1 = self.ty
        alpha2 = element2.rotation_angle
        x2 = element2.tx
        y2 = element2.ty
        alpha_subs = alpha1 - alpha2
        return Se2A(alpha_subs, x1 - x2, y1 - y2).quot

    def __rmul__(self, scalar):
        """
        Alternative for the scalar product
        """
        tx = self.tx
        ty = self.ty
        alpha = self.rotation_angle
        return Se2A(scalar * alpha, scalar * tx, scalar * ty).quot

    def scalarprod(self, const):
        """
        :param const: scalar float value
        :return: const*element as scalar product in the Lie algebra
        """
        alpha = self.rotation_angle
        x = self.tx
        y = self.ty
        return Se2A(const * alpha, const * x, const * y)

    def norm(self, norm_type='standard', lamb=1):
        """
        norm(self, typology, lamb for the standard norm type)

        norm_type get the type of the norm we want to deal with:
        norm_type in {'', 'vector_len'}

        :param norm_type:
        :param lamb:
        :return:
        """
        if norm_type == 'lamb':
            ans = sqrt(self.rotation_angle**2 + lamb*(self.tx**2 + self.ty**2))  # lamb has to be modified in aux_func
        elif norm_type == 'translation':
            ans = sqrt(self.tx ** 2 + self.ty ** 2)
        elif norm_type == 'fro':
            ans = sqrt(2*self.rotation_angle**2 + self.tx ** 2 + self.ty ** 2)
        else:
            ans = -1
        return ans


def se2a_randomgen(intervals=(0, 10), norm_type='fro', lamb=1):
    """
    Montecarlo sampling to build a random element in Se2A.
    it uses 3 possible norm_type, lamb, translation, or fro

    :param intervals:
    :param norm_type:
    :param lamb: weight of the radius: sqrt(theta**2 + lamb * rho**2)
    :return: se2_a(theta, tx, ty) such that
    a < se2_a(theta, tx, ty).norm(norm_type) < b
    """
    # montecarlo sampling to get a < se2_a(theta, tx, ty).norm(norm_type) < b
    norm_belongs_to_interval = False
    num_attempts = 0
    a = intervals[0]
    b = intervals[1]

    theta, tx, ty = 0, 0, 0

    if norm_type == 'lamb':

        while not norm_belongs_to_interval and num_attempts < 1000:
            num_attempts += 1
            theta = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
            rho = uniform(a, b)

            if a <= sqrt(theta**2 + lamb * rho**2) <= b:
                # keep theta, compute tx ty according to rho and a new eta
                eta = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
                tx = rho * cos(eta)
                ty = rho * sin(eta)

                norm_belongs_to_interval = True

    elif norm_type == 'translation':

        # we pick rho to satisfy the condition a < se2_a(theta, tx, ty).norm(norm_type) < b
        # where norm is the the norm of the translation.
        # no Montecarlo is needed.
        theta = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
        rho = uniform(a, b)
        eta = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
        tx = rho * cos(eta)
        ty = rho * sin(eta)

    elif norm_type == 'fro':

        while not norm_belongs_to_interval and num_attempts < 1000:
            num_attempts += 1
            theta = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
            rho = uniform(a, b)

            if a <= sqrt(2*theta**2 + rho**2) <= b:
                # keep theta, compute tx ty according to rho and a new eta
                eta = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
                tx = rho * cos(eta)
                ty = rho * sin(eta)

                norm_belongs_to_interval = True

    if num_attempts == 1000 and not norm_belongs_to_interval:
        raise Exception("randomgen_standard_norm in se2_g with lambda = 1: safety recursion number has been reached"
                        "montecarlo method has failed in finding an element after 1000 attempt.")

    else:
        return Se2A(theta, tx, ty)


def is_a_matrix_in_se2a(m_input, relax=False):
    ans = False
    if type(m_input) == np.ndarray and m_input.shape == (3, 3):
        m = m_input.copy()
        if relax:
            m = np.around(m, 6)
        if m[0, 0] == m[1, 1] == 0 and m[1, 0] == -m[0, 1] and m[2, 0] == m[2, 1] == m[2, 2] == 0:
            ans = True
    return ans


def se2a_lie_bracket(element1, element2):
    """
    lie_bracket(A,B) \n
    :param element1 :
    :param element2 : elements of se2 already in the quotient
    :return: their Lie bracket
    Note: no need to put quotient. The resultant angle is zero.
    """
    # if str(type(element1))[-5:] != str(type(element2))[-5:] != 'se2_a':
    #    raise TypeError('warning: wrong input data format, 2 se2_a elements expected.')

    element1 = element1.quot
    element2 = element2.quot
    alpha1 = element1.rotation_angle
    alpha2 = element2.rotation_angle
    x1 = element1.tx
    y1 = element1.ty
    x2 = element2.tx
    y2 = element2.ty
    return Se2A(0, alpha2 * y1 - alpha1 * y2, alpha1 * x2 - alpha2 * x1)


def se2a_lie_multi_bracket(l):
    """
    lie_bracket(l) \n
    :param l: L list of element of se2
    :return: [L0 ,[L1,[L2, [... [L(n-1),Ln]...]
    """
    if len(l) <= 1:
        return 0
    elif len(l) == 2:
        return se2a_lie_bracket(l[0], l[1])
    else:
        le = len(l)
        return se2a_lie_multi_bracket(l[0:le - 2] + [se2a_lie_bracket(l[le - 2], l[le - 1])])


def matrix2se2a(a, eat_em_all=False):
    """
    matrix2se2(a, relax = False) \n
    :param a: matrix of se(2) np.array([[0,0,0],[tx,0,-theta],[ty,theta,0]])
            optional relax, if it relax the input condition, allowing any kind of matrix
    :param eat_em_all: if True, any kind of matrix can be utilized and converted into se2, cutting the extra info.
    :return: corresponding element in SE2 .
    """
    if not eat_em_all and not is_a_matrix_in_se2a(a):
        raise Exception("matrix2se2_a in se2_a: the inserted element is not a matrix in se2_a  ")
    else:
        theta = a[1, 0]
        x = a[0, 2]
        y = a[1, 2]
        return Se2A(theta, x, y)


def list2se2a(a):
    if not type(a) == list or not len(a) == 3:
        raise TypeError("list2se2_g in se2_g: list of dimension 3 expected")
    return Se2A(a[0], a[1], a[2])


def se2a_exp(element):
    """
    exp(element) \n
    algebra exponential
    :param element: instance of se2
    :return: corresponding element in Lie group SE2, se2_g.matrix2se2_g(lin.expm(m_element), True)
    """
    element = element.quot
    theta = element.rotation_angle
    v1 = element.tx
    v2 = element.ty
    esp = np.abs(np.spacing([0]))[0]
    if abs(theta) <= 10*esp:
        factor1 = 1 - (theta**2)/6.0
        factor2 = theta/2.0
    else:
        factor1 = sin(theta)/theta
        factor2 = (1 - cos(theta))/theta
    x1 = factor1*v1 - factor2*v2
    x2 = factor2*v1 + factor1*v2

    return Se2G(theta, x1, x2)


def se2a_exp_matrices(m, eat_em_all=True):
    """
    exp_m(m) \n
    algebra exponential for matrices
    :param m: instance of se2 in matrix form
    :param eat_em_all: if any matrix can be its input. It will be forced to be exponentiated, cutting the extra info.
    :return: corresponding element in Lie group SE2, matrix form again
    """
    if not eat_em_all and not is_a_matrix_in_se2a(m):
        raise Exception("exp_for_matrices in se2_a: the inserted element is not a matrix in se2_a  ")

    # first step is to quotient the input element (if the input is a sum)
    ans = se2a_exp(Se2A(m[1, 0], m[0, 2], m[1, 2]))
    return ans.get_matrix


def se2a_lie_bracket_for_matrices(m1, m2, eat_em_all=True):
    """
    Compute lie bracket for matrices
    :param m1:
    :param m2:
    :param eat_em_all: how strict we are on input!
    :return:
    """
    if not eat_em_all:
        if not (is_a_matrix_in_se2a(m1) and is_a_matrix_in_se2a(m2)):
            raise Exception("bracket_for_matrices in se2_a: the inserted element is not a matrix in se2_a  ")

    return m1*m2 - m2*m1


def se2a_lie_multi_bracket_for_matrices(l):
    """
    lie_bracket(L) \n
    :param l: list of matrices
    :return: [L0 ,[L1,[L2, [... [L(n-1),Ln]...]
    """
    if len(l) <= 1:
        return 0
    elif len(l) == 2:
        return se2a_lie_bracket_for_matrices(l[0], l[1])
    else:
        num = len(l)
        return se2a_lie_multi_bracket_for_matrices(l[0:num - 2] + [se2a_lie_bracket(l[num - 2], l[num - 1])])


""" Lie group """


class Se2G(object):
    """
    Class for group of SE(2) Lie group of rotations and translation in 2 dimensions.
    To respect camel case convention and avoid confusion class for SE(2) is called Se2G.
    """

    def __init__(self, theta, tx, ty):
        self.rotation_angle = mod_pipi(theta)
        self.tx = tx
        self.ty = ty

    def __get_matrix__(self):
        a1 = [cos(self.rotation_angle), -sin(self.rotation_angle), self.tx]
        a2 = [sin(self.rotation_angle),  cos(self.rotation_angle), self.ty]
        a3 = [0, 0, 1]
        return np.array([a1, a2, a3])

    get_matrix = property(__get_matrix__)

    def __get_restricted__(self):
        return [self.rotation_angle, self.tx, self.ty]

    get = property(__get_restricted__)

    def __mul__(self, element2):
        x1 = self.tx
        x2 = element2.tx
        y1 = self.ty
        y2 = element2.ty
        theta_1 = mod_pipi(self.rotation_angle)
        c1 = cos(theta_1)
        s1 = sin(theta_1)

        alpha = mod_pipi(self.rotation_angle + element2.rotation_angle)
        x = x1 + x2 * c1 - y2 * s1
        y = y1 + x2 * s1 + y2 * c1

        return Se2G(alpha, x, y)

    def inv(self):
        alpha = self.rotation_angle
        xv = self.tx
        yv = self.ty
        xn = - cos(alpha)*xv - sin(alpha)*yv
        yn = + sin(alpha)*xv - cos(alpha)*yv
        return Se2G(-alpha, xn, yn)

    def norm(self, norm_type='standard', lamb=1):
        """
        norm(self, typology, order = None, axes = None)

        norm_type get the type of the norm we want to deal with:
        norm_type in {'', 'vector_len'}

        """
        if norm_type == 'standard' or norm_type == 'group_norm':
            ans = sqrt(self.rotation_angle**2 + lamb*(self.tx**2 + self.ty**2))
        elif norm_type == 'translation':
            ans = sqrt(self.tx ** 2 + self.ty ** 2)
        elif norm_type == 'fro':
            ans = sqrt(3 + self.tx ** 2 + self.ty ** 2)
        else:
            ans = -1
        return ans


def se2g_randomgen_custom(interval_theta=(), interval_tx=(), interval_ty=(), avoid_zero_rotation=True):
    """
    GIGO method.
    :param interval_theta:  (interval_theta[0], interval_theta[1])
    :param interval_tx:     (interval_tx[0], interval_tx[1])
    :param interval_ty:     (interval_ty[0], interval_ty[1])
    :param avoid_zero_rotation:
    :return:
    se2_g(uniform(interval_theta[0], interval_theta[1]),
          uniform(interval_tx[0], interval_tx[1]),
          uniform(interval_ty[0], interval_ty[1]) )
    """
    # avoid theta = zero

    theta = uniform(interval_theta[0], interval_theta[1])

    if avoid_zero_rotation:
        if interval_theta[0] < 0 < interval_theta[1]:
            epsilon = 0.001  # np.spacing(0)
            theta = np.random.choice([uniform(interval_theta[0], 0 - epsilon), uniform(0 + epsilon, interval_theta[1])])

    if len(interval_tx) == 1:
        tx = interval_tx[0]
    else:
        tx = uniform(interval_tx[0], interval_tx[1])

    if len(interval_ty) == 1:
        ty = interval_ty[0]
    else:
        ty = uniform(interval_ty[0], interval_ty[1])

    return Se2G(theta, tx, ty)


def se2g_randomgen_custom_center(interval_theta=(-np.pi / 2, np.pi / 2), omega=(1, 6), avoid_zero_rotation=True,
                                 epsilon_zero_avoidance=0.001):
    """
    An element of SE(2) defines a rotation (from SO(2)) away from the origin.
    The center of the rotation is the fixed point of the linear map.
    This method provides an element of SE(2), random, with rotation parameter sampled over a uniform distribution
    in the interval interval_theta, and with center of rotation in the squared subset of the cartesian plane
    (x0,x1)x(y0,y1)
    :param interval_theta:
    :param omega:
    :param avoid_zero_rotation:
    :param epsilon_zero_avoidance:
    :return:
    """

    theta  = uniform(interval_theta[0], interval_theta[1])

    if avoid_zero_rotation:
        if interval_theta[0] < 0 < interval_theta[1]:
            epsilon = epsilon_zero_avoidance  # np.spacing(0)
            theta = np.random.choice([uniform(interval_theta[0], 0 - epsilon), uniform(0 + epsilon, interval_theta[1])])

    x_c  = uniform(omega[0], omega[1])
    y_c  = uniform(omega[0], omega[1])

    tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
    ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

    return Se2G(theta, tx, ty)


def se2g_randomgen(intervals=(), lamb=1):
    if len(intervals) == 2:
        a = intervals[0]
        b = intervals[1]
        if b < a or a < 0 or b < 0:
            raise Exception("randomgen_standard_norm in se2_g  "
                            "given interval (a,b) must have a < b for a, b positive ")
    elif len(intervals) == 0:
        a = 0
        b = 10
    else:
        raise Exception("randomgen_standard_norm in se2_g: "
                        "the list of intervals inserted must have dimension 2 or 0")

    if lamb < 0:
        raise Exception("randomgen_standard_norm in se2_g with lambda < 0: "
                        "negative lambda is not accepted. ")
    elif lamb == 0:
        if a > np.pi:
            raise Exception("randomgen_standard_norm in se2_g with lambda = 0: "
                            "intervals (a,b) and (-pi, pi) can not be disjoint ")
        else:
            a_theta_pos = a
            b_theta_pos = min(b, np.pi)
            a_theta_neg = -a
            b_theta_neg = max(-b, -np.pi + np.abs(np.spacing(-np.pi)))
            theta = np.random.choice([uniform(a_theta_pos, b_theta_pos), uniform(a_theta_neg, b_theta_neg)])
            rho = uniform(0, 10)
            eta = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
            tx = rho * cos(eta)
            ty = rho * sin(eta)
    else:
        a_theta = max(-a, -np.pi + np.abs(np.spacing(-np.pi)))
        b_theta = min(a, np.pi)
        theta = uniform(a_theta, b_theta)
        rho = uniform(sqrt((a**2 - theta**2)/lamb), sqrt((b**2 - theta**2)/lamb))
        eta = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
        tx = rho * cos(eta)
        ty = rho * sin(eta)
    return Se2G(theta, tx, ty)


def se2g_randomgen_translation(intervals=()):
    if len(intervals) == 2:
        a = intervals[0]
        b = intervals[1]
        if b < a or a < 0 or b < 0:
            raise Exception("randomgen_standard_norm in se2_g with: "
                            "given interval (a,b) must have a < b for a, b positive ")
    elif len(intervals) == 0:
        a = 0
        b = 10
    else:
        raise Exception("randomgen_translation_norm in se2_g: "
                        "the list of intervals inserted must have dimension 2 or 0")
    theta = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
    rho = uniform(a, b)
    eta = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
    tx = rho * cos(eta)
    ty = rho * sin(eta)
    return Se2G(theta, tx, ty)


def se2g_randomgen_fro(intervals=()):
    if len(intervals) == 2:
        a = intervals[0]
        b = intervals[1]
        if b < a or a < sqrt(3):
            raise Exception("randomgen_fro_norm in se2_g with: given interval (a,b) "
                            "must have a < b for a, b positive greater than sqrt(3) ")
    elif len(intervals) == 0:
        a = sqrt(3)
        b = 10
    else:
        raise Exception("randomgen_translation_norm in se2_g: "
                        "the list of intervals inserted must have dimension 2 or 0")
    theta = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
    rho = uniform(sqrt(round(a**2 - 3, 15)), sqrt(round(b**2 - 3, 15)))
    eta = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
    tx = rho * cos(eta)
    ty = rho * sin(eta)
    return Se2G(theta, tx, ty)


def is_a_matrix_in_se2g(m_input, relax=False):
    ans = False
    if type(m_input) == np.ndarray:
        m = m_input.copy()
        if relax:
            m = np.around(m, 6)
        if m.shape == (3, 3) and m[0, 0] == m[1, 1] and m[1, 0] == -m[0, 1] \
                and m[2, 0] == m[2, 1] == 0 and m[2, 2] == 1:
            ans = True
    return ans


def matrix2se2g(A, eat_em_all=False):
    """
    matrix2se2_g(M) \n
    :param: matrix of SE(2) np.array([[1,0],[t,R]]). t translation (in IR^(2x1)), R rotation (in SO(2))
            optional relax, if it relax the input condition, allowing any kind of matrix
            Relax, if we want some relaxation in the input.
    :return: corresponding element in SE2 .
    """
    if not eat_em_all and not is_a_matrix_in_se2g(A):
        raise Exception("matrix2se2_g in se2_g: the inserted element is not a matrix in se2_g  ")
    else:
        theta = np.arctan2(A[1, 0], A[0, 0])
        x = A[0, 2]
        y = A[1, 2]
        return Se2G(theta, x, y)


def list2se2g(a):
    if not type(a) == list or not len(a) == 3:
        raise TypeError("list2se2_g in se2_g: list of dimension 3 expected")
    return Se2G(a[0], a[1], a[2])


def se2g_log(element):
    """
    log(element) \n
    group logarithm
    :param: instance of SE2
    :return: corresponding element in Lie algebra se2
    """
    theta = mod_pipi(element.rotation_angle)
    v1 = element.tx
    v2 = element.ty
    c = cos(theta)
    prec = np.abs(np.spacing([0.0]))[0]
    if abs(c - 1.0) <= 10*prec:
        x1 = v1
        x2 = v2
        theta = 0
    else:
        factor = (theta / 2.0) * sin(theta) / (1.0 - c)
        x1 = factor * v1 + (theta / 2.0) * v2
        x2 = factor * v2 - (theta / 2.0) * v1
    return Se2A(theta, x1, x2)
