import copy

import numpy as np
from scipy import integrate
from scipy.linalg import expm
from scipy.misc import factorial as fact

from calie.aux import matrices
from calie.operations import jacobians as jac
from calie.fields import queries as qr
from calie.fields import compose as cp


class LieExp:
    def __init__(self):
        self.s_i_o = 3

        self.dimension = None
        self.omega = None
        self.vf = None
        self.phi = None

        self.num_steps = None

    def initialise_input(self, input_vf):
        """ Initialise class variable given an input vf """
        self.dimension = qr.check_is_vf(input_vf)
        self.omega = qr.get_omega(input_vf)

        self.vf = copy.deepcopy(input_vf)
        self.phi = np.zeros_like(self.vf)

    def initialise_number_of_steps(self, input_num_steps=None, input_pix_dims=None):
        """ Automatic steps selector """
        if input_num_steps is None:
            norm = np.linalg.norm(self.vf, axis=self.dimension - 1)
            max_norm = np.max(norm[:])

            if max_norm < 0:
                raise ValueError('Maximum norm is invalid.')
            if max_norm == 0:
                return self.phi

            if input_pix_dims is None:
                self.num_steps = max([0, np.ceil(np.log2(max_norm / 0.5)).astype('int')]) + 3
            else:
                min_size = np.min(input_pix_dims[input_pix_dims > 0])
                self.num_steps = max([0, np.ceil(np.log2(max_norm / (min_size / 2))).astype('int')]) + 3

        elif input_num_steps is 'test_method':
            norm_vf = np.linalg.norm(self.vf, axis=self.vf.ndim - 1)
            max_norm = np.max(norm_vf)
            toll = 1e-3
            k = 10
            while max_norm / fact(k) > toll:
                k += 1
                self.num_steps = k
            print('automatic steps selector for series method: ' + str(k))

        elif isinstance(input_num_steps, int):
            self.num_steps = input_num_steps

        else:
            raise IOError

    # Numerical solvers:

    def scaling_and_squaring(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ classical scaling and squaring """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        # (1)
        init = 1 << self.num_steps
        self.phi = self.vf / float(init)

        # (2)
        for _ in range(0, self.num_steps):
            self.phi = cp.lagrangian_dot_lagrangian(self.phi, self.phi, s_i_o=self.s_i_o)

        return self.phi

    def gss_ei(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ generalised scaling and squaring exponetial integrator """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        # (1)
        init = 1 << self.num_steps
        self.phi = self.vf / init

        # (1.5)
        jv = jac.compute_jacobian(self.phi)

        if self.dimension == 2:

            v_matrix = np.array([0.0] * 3 * 3).reshape([3, 3])

            for x in range(self.phi.shape[0]):
                for y in range(self.phi.shape[1]):
                    # skew symmetric part
                    v_matrix[0:2, 0:2] = jv[x, y, 0, 0, :].reshape([2, 2])
                    # translational part
                    v_matrix[0, 2], v_matrix[1, 2] = self.phi[x, y, 0, 0, 0:2]  # + \
                    # jv[x, y, 0, 0, :].reshape([2, 2]).dot([x, y])

                    # translational part of the exp is the answer:
                    self.phi[x, y, 0, 0, :] = expm(v_matrix)[0:2, 2]

        elif self.dimension == 3:

            v_matrix = np.array([0.0] * 4 * 4).reshape([4, 4])

            for x in range(self.phi.shape[0]):
                for y in range(self.phi.shape[1]):
                    for z in range(self.phi.shape[2]):
                        # skew symmetric part
                        v_matrix[0:3, 0:3] = jv[x, y, z, 0, :].reshape([3, 3])

                        # translation part
                        v_matrix[0, 3], v_matrix[1, 3], v_matrix[2, 3] = self.phi[x, y, z, 0, 0:3]

                        self.phi[x, y, z, 0, :] = expm(v_matrix)[0:3, 3]

        # (2)
        for _ in range(0, self.num_steps):
            self.phi = cp.lagrangian_dot_lagrangian(self.phi, self.phi, s_i_o=self.s_i_o)

        return self.phi

    def gss_ei_mod(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ generalised scaling and squaring exponential integrators modified """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        # (1) copy the reduced v in phi, future solution of the ODE.
        init = 1 << self.num_steps
        self.phi = self.vf / init

        # (1.5)
        jv = jac.compute_jacobian(self.phi)

        if self.dimension == 2:

            for x in range(self.phi.shape[0]):
                for y in range(self.phi.shape[1]):
                    j = jv[x, y, 0, 0, :].reshape([2, 2])
                    tr = self.phi[x, y, 0, 0, 0:2]
                    j_tr = j.dot(tr)
                    self.phi[x, y, 0, 0, :] = tr + 0.5 * j_tr  # + 1/6. * J.dot(J_tr)

        elif self.dimension == 3:

            for x in range(self.phi.shape[0]):
                for y in range(self.phi.shape[1]):
                    for z in range(self.phi.shape[2]):
                        j = jv[x, y, z, 0, :].reshape([3, 3])
                        tr = self.phi[x, y, z, 0, 0:3]
                        j_tr = j.dot(tr)
                        self.phi[x, y, z, 0, :] = tr + 0.5 * j_tr  # + 1/6. * j.dot(j_tr)

        # (2)
        for _ in range(0, self.num_steps):
            self.phi = cp.lagrangian_dot_lagrangian(self.phi, self.phi, s_i_o=self.s_i_o)

        return self.phi

    def gss_aei(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ generalised scaling and squaring approximated exponential integrators """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        # (1)
        if self.num_steps == 0:
            self.phi = self.vf
        else:
            init = 1 << self.num_steps
            self.phi = self.vf / init

        # (1.5)  phi = 1 + v + 0.5jac*v
        jv = np.squeeze(jac.compute_jacobian(self.phi))
        v_sq = np.squeeze(self.phi)
        new_shape = list(self.omega) + [1] * (4 - self.dimension) + [self.dimension]
        jv_prod_v = matrices.matrix_vector_field_product(jv, v_sq).reshape(new_shape)

        self.phi += 0.5 * jv_prod_v

        # (2)
        for _ in range(0, self.num_steps):
            self.phi = cp.lagrangian_dot_lagrangian(self.phi, self.phi, s_i_o=self.s_i_o)

        return self.phi

    def midpoint(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ midpoint method """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        # (1, 2)
        if self.num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / self.num_steps

        for _ in range(self.num_steps):
            phi_tilda = self.phi + (h / 2) * cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False)
            self.phi += h * cp.lagrangian_dot_lagrangian(self.vf, phi_tilda, s_i_o=self.s_i_o, add_right=False)

        return self.phi

    def series(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ Series method """
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        self.phi = np.copy(self.vf)  # phi is initialised to vf not to zero.

        for k in range(2, self.num_steps):
            jac_v = jac.iterative_jacobian_product(self.vf, k)
            self.phi = self.phi[...] + jac_v[...] / fact(k)

        return self.phi

    def series_mod(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """  """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        jac_v = copy.deepcopy(self.vf)
        self.phi = np.copy(self.vf)  # phi is initialised to vf not to zero.

        for k in range(1, input_num_steps):
            jac_v = jac.jacobian_product(jac_v, self.vf)
            self.phi = self.phi[...] + jac_v[...] / fact(k)

        return self.phi

    def euler(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """  """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        if self.num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / self.num_steps

        for _ in range(self.num_steps):
            self.phi += h * cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False)

        return self.phi

    def euler_aei(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ euler with approximated exponential integrators """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        self.vf = self.vf / self.num_steps

        jv = np.squeeze(jac.compute_jacobian(self.vf))
        v_sq = np.squeeze(self.vf)
        new_shape = list(self.omega) + [1] * (4 - self.dimension) + [self.dimension]
        jv_prod_v = matrices.matrix_vector_field_product(jv, v_sq).reshape(new_shape)

        self.vf += 0.5 * jv_prod_v

        for _ in range(self.num_steps):
            self.phi = cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o)

        return self.phi

    def euler_mod(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ Euler modified method """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        if self.num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / self.num_steps

        for _ in range(self.num_steps):

            phi_tilda = self.phi + h * cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False)

            self.phi += (h/2) * (cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False) +
                                 cp.lagrangian_dot_lagrangian(self.vf, phi_tilda, s_i_o=self.s_i_o, add_right=False))

        return self.phi

    def heun(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ Heun method """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        if self.num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / self.num_steps

        for i in range(self.num_steps):

            psi_1 = self.phi + h * (2. / 3) * cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False)

            psi_2 = cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False) + \
                                                 3 * cp.lagrangian_dot_lagrangian(self.vf, psi_1, s_i_o=self.s_i_o, add_right=False)

            self.phi += (h / 4) * psi_2

        return self.phi

    def heun_mod(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ Heun modified method """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        if self.num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / self.num_steps
        for i in range(self.num_steps):

            psi_1 = self.phi + (h / 3) * cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False)

            psi_2 = self.phi + h * (2. / 3) * cp.lagrangian_dot_lagrangian(self.vf, psi_1, s_i_o=self.s_i_o, add_right=False)

            psi_3 = cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False) + \
                                              3 * cp.lagrangian_dot_lagrangian(self.vf, psi_2, s_i_o=self.s_i_o, add_right=False)

            self.phi += (h / 4) * psi_3

        return self.phi

    def rk4(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ Runge Kutta 4 """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        if self.num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / self.num_steps

        for _ in range(self.num_steps):

            r_1 = h * cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False)

            psi_1 = self.phi + .5 * r_1
            r_2 = h  * cp.lagrangian_dot_lagrangian(self.vf, psi_1, s_i_o=self.s_i_o, add_right=False)

            psi_2 = self.phi + .5 * r_2
            r_3 = h * cp.lagrangian_dot_lagrangian(self.vf, psi_2, s_i_o=self.s_i_o, add_right=False)

            psi_3 = self.phi + r_3
            r_4 = h  * cp.lagrangian_dot_lagrangian(self.vf, psi_3, s_i_o=self.s_i_o, add_right=False)

            self.phi += (1. / 6) * (r_1 + 2 * r_2 + 2 * r_3 + r_4)

        return self.phi

    def gss_rk4(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ Generalised scaling and squaring runge kutta method  """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        norm = np.linalg.norm(self.vf, axis=self.vf.ndim - 1)
        max_norm = np.max(norm[:])

        if max_norm == 0:
            return self.phi

        if self.num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / self.num_steps

        # (1)
        init = 1 << self.num_steps
        self.vf = self.vf / init

        # (1.5)
        for _ in range(self.num_steps):
            r_1 = h * cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False)

            psi_1 = self.phi + .5 * r_1
            r_2 = h * cp.lagrangian_dot_lagrangian(self.vf, psi_1, s_i_o=self.s_i_o, add_right=False)

            psi_2 = self.phi + .5 * r_2
            r_3 = h * cp.lagrangian_dot_lagrangian(self.vf, psi_2, s_i_o=self.s_i_o, add_right=False)

            psi_3 = self.phi + r_3
            r_4 = h * cp.lagrangian_dot_lagrangian(self.vf, psi_3, s_i_o=self.s_i_o, add_right=False)

            self.phi += (1. / 6) * (r_1 + 2 * r_2 + 2 * r_3 + r_4)

        # (2)
        for _ in range(self.num_steps):
            self.phi = cp.lagrangian_dot_lagrangian(self.phi, self.phi, s_i_o=self.s_i_o, add_right=True)

        return self.phi

    def trapeziod_euler(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """ Trapezoid Euler """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        if self.num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / self.num_steps

        for _ in range(0, self.num_steps):
            # euler
            phi_tilda = self.phi + h * cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False)

            self.phi += .5 * h * (cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False) +
                                  cp.lagrangian_dot_lagrangian(self.vf, phi_tilda, s_i_o=self.s_i_o, add_right=False))

        return self.phi

    def trapezoid_midpoint(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """  """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        if self.num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / self.num_steps

        for _ in range(0, self.num_steps):
            # midpoint
            phi_tilda_tilda = self.phi + (h / 2) * cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False)
            phi_tilda = self.phi + h * cp.lagrangian_dot_lagrangian(self.vf, phi_tilda_tilda, s_i_o=self.s_i_o, add_right=False)

            self.phi += .5 * h * (cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False) +
                                  cp.lagrangian_dot_lagrangian(self.vf, phi_tilda, s_i_o=self.s_i_o, add_right=False))

        return self.phi

    def gss_trapezoid_euler(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """  """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        if self.num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / self.num_steps

        # (1)
        init = 1 << self.num_steps
        self.vf = self.vf / init

        # (1.5)
        for _ in range(0, self.num_steps):
            tilda_phi = self.phi + h * cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False)
            self.phi += .5 * h * (cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False) +
                             cp.lagrangian_dot_lagrangian(self.vf, tilda_phi, s_i_o=self.s_i_o, add_right=False))

        # (2)
        for _ in range(0, self.num_steps):
            self.phi = cp.lagrangian_dot_lagrangian(self.phi, self.phi, s_i_o=self.s_i_o, add_right=True)

        return self.phi

    def gss_trapezoid_midpoint(self, input_vf, input_num_steps=None, input_pix_dims=None):
        """  """
        # (0)
        self.initialise_input(input_vf)
        self.initialise_number_of_steps(input_num_steps=input_num_steps, input_pix_dims=input_pix_dims)

        if self.num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / self.num_steps
        # TODO correct h for init value.

        # (1)
        init = 1 << self.num_steps
        self.vf = self.vf / init

        # (1.5)
        for _ in range(0, self.num_steps):
            phi_tilda_tilda = self.phi + (h / 2) * cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False)
            phi_tilda = self.phi + h * cp.lagrangian_dot_lagrangian(self.vf, phi_tilda_tilda, s_i_o=self.s_i_o, add_right=False)

            self.phi += .5 * h * (cp.lagrangian_dot_lagrangian(self.vf, self.phi, s_i_o=self.s_i_o, add_right=False) +
                                  cp.lagrangian_dot_lagrangian(self.vf, phi_tilda, s_i_o=self.s_i_o, add_right=False))

        # (2)
        for _ in range(0, self.num_steps):
            self.phi = cp.lagrangian_dot_lagrangian(self.phi, self.phi, s_i_o=self.s_i_o, add_right=True)

        return self.phi

    def scipy_pointwise(self,
                        input_vf,
                        integrator='vode',
                        method='bdf',
                        max_steps=7,
                        interpolation_method='cubic',
                        passepartout=3,
                        return_integral_curves=False,
                        verbose=False):
        """
        Compute the exponential of this velocity field using scipy libraries.
        :param input_vf:
        :param integrator: vode, zvode, lsoda, dopri5, dopri853, see scipy documentations:
            http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.integrate.ode.html
        :param method: adams (non-stiff) or bdf (stiff)
        :param interpolation_method:
        :param max_steps: maximum steps of integrations. If less steps are required, it prints a warning message.
        :param passepartout:
        :param return_integral_curves:
        :param verbose:
        :param : It returns a displacement phi, element of the class disp, as
                    phi(x) = psi(x) - x
                where psi is the chosen integrator.

        """
        self.initialise_input(input_vf)

        assert self.dimension == 2

        flows_collector = []

        # transform v in a function suitable for ode library :
        def vf_function(t, x):
            return list(cp.one_point_interpolation(self.vf, point=x, method=interpolation_method))

        t0, t_n = 0, 1
        dt = (t_n - t0) / float(max_steps)

        r = integrate.ode(vf_function).set_integrator(integrator, method=method, max_step=max_steps)

        for i in range(0 + passepartout, self.omega[0] - passepartout + 1):
            for j in range(0 + passepartout, self.omega[1] - passepartout + 1):  # cycle on the point of the grid.

                y = []
                r.set_initial_value([i, j], t0).set_f_params()  # initial conditions are point on the grid
                while r.successful() and r.t + dt < t_n:
                    r.integrate(r.t + dt)
                    y.append(r.y)

                # flow of the svf at the point [i,j]
                fl = np.array(np.real(y))

                # subtract id on the run to return a displacement.
                if fl.shape[0] > 0:
                    self.phi[i, j, 0, 0, :] = fl[fl.shape[0] - 1, :] - np.array([i, j])
                else:
                    self.phi[i, j, 0, 0, :] = - np.array([i, j])

                if verbose:
                    print('Integral curve at grid point ' + str([i, j]) + ' is computed.')

                # In some cases as critical points, or when too closed to the closure of the domain
                # the number of steps can be reduced by the algorithm.
                if fl.shape[0] < max_steps - 2 and verbose:  # the first step is not directly considered
                    print("--------")
                    print("Warning!")  # steps jumped for the point
                    print("--------")

                if return_integral_curves:
                    flows_collector += [fl]

        if return_integral_curves:
            return self.phi, flows_collector
        else:
            return self.phi
