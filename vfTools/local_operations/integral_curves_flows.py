'''
def exponential_scipy(self,
                      integrator='vode',
                      method='bdf',
                      max_steps=7,
                      interpolation_method='cubic',
                      passepartout=3,
                      return_integral_curves=False,
                      verbose=False):
    """
    Compute the exponential of this velocity field using scipy libraries.
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
    v = copy.deepcopy(SVF.from_field(self.field, header=self.nib_image.get_header()))
    phi = copy.deepcopy(SDISP.generate_zero(shape=self.shape,
                                            header=self.nib_image.get_header(),
                                            affine=self.mm_2_voxel))

    flows_collector = []

    # transform v in a function suitable for ode library :
    def vf(t, x):
        return list(v.one_point_interpolation(point=x, method=interpolation_method))

    t0, t_n = 0, 1
    dt = (t_n - t0)/float(max_steps)

    r = ode(vf).set_integrator(integrator, method=method, max_step=max_steps)

    for i in range(0 + passepartout, phi.vol_ext[0] - passepartout + 1):
        for j in range(0 + passepartout, phi.vol_ext[1] - passepartout + 1):  # cycle on the point of the grid.

            y = []
            r.set_initial_value([i, j], t0).set_f_params()  # initial conditions are point on the grid
            while r.successful() and r.t + dt < t_n:
                r.integrate(r.t+dt)
                y.append(r.y)

            # flow of the svf at the point [i,j]
            fl = np.array(np.real(y))

            # subtract id on the run to return a displacement.
            if fl.shape[0] > 0:
                phi.field[i, j, 0, 0, :] = fl[fl.shape[0]-1, :] - np.array([i, j])
            else:
                phi.field[i, j, 0, 0, :] = - np.array([i, j])

            if verbose:
                print 'Integral curve at grid point ' + str([i, j]) + ' is computed.'

            # In some cases as critical points, or when too closed to the closure of the domain
            # the number of steps can be reduced by the algorithm.
            if fl.shape[0] < max_steps - 2 and verbose:  # the first step is not directly considered
                print "--------"
                print "Warning!"  # steps jumped for the point
                print "--------"

            if return_integral_curves:
                flows_collector += [fl]

    if return_integral_curves:
        return phi, flows_collector
    else:
        return phi

'''