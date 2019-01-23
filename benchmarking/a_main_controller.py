from collections import OrderedDict

from calie.operations import lie_exp


spline_interpolation_order = 3

steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

l_exp = lie_exp.LieExp()
l_exp.s_i_o = spline_interpolation_order

# name, active num_steps_default (None is automatic) colour  line-style   marker, [sub-options]]

methods = OrderedDict()

methods.update({l_exp.scaling_and_squaring.__name__       : [l_exp.scaling_and_squaring,   True,    7,    'b',      '-',     '+', []]})
methods.update({l_exp.gss_ei.__name__                     : [l_exp.gss_ei,                 True,    7,    'b',      '-',     '+', []]})
methods.update({l_exp.gss_ei_mod.__name__                 : [l_exp.gss_ei_mod,             True,    7,    'b',     '--',     '+', []]})
methods.update({l_exp.gss_aei.__name__                    : [l_exp.gss_aei,                True,    7,    'r',      '-',     '.', []]})
methods.update({l_exp.midpoint.__name__                   : [l_exp.midpoint,               True,    7,    'r',     '--',     'x', []]})
methods.update({l_exp.series.__name__                     : [l_exp.series,                 False,   10,   'b',      '-',     '*', []]})
methods.update({l_exp.series_mod.__name__                 : [l_exp.series_mod,             False,   10,   'c',      '-',     '.', []]})
methods.update({l_exp.euler.__name__                      : [l_exp.euler,                  True,    40,   'g',      '-',     '>', []]})
methods.update({l_exp.euler_aei.__name__                  : [l_exp.euler_aei,              True,    40,   'm',     '--',     '>', []]})
methods.update({l_exp.euler_mod.__name__                  : [l_exp.euler_mod,              True,    40,   'm',      '-',     '>', []]})
methods.update({l_exp.heun.__name__                       : [l_exp.heun,                   True,    40,   'm',     '--',     '>', []]})
methods.update({l_exp.heun_mod.__name__                   : [l_exp.heun_mod,               True,    10,   'k',      '-',     '.', []]})
methods.update({l_exp.rk4.__name__                        : [l_exp.rk4,                    True,    10,   'k',     '--',     '.', []]})
methods.update({l_exp.gss_rk4.__name__                    : [l_exp.gss_rk4,                True,    10,   'y',     '--',     'x', []]})
methods.update({l_exp.trapeziod_euler.__name__            : [l_exp.trapeziod_euler,        True,    10,   'y',     '--',     'x', []]})
methods.update({l_exp.trapezoid_midpoint.__name__         : [l_exp.trapezoid_midpoint,     True,    10,   'y',     '--',     'x', []]})
methods.update({l_exp.gss_trapezoid_euler.__name__        : [l_exp.gss_trapezoid_euler,    True,    10,   'y',     '--',     'x', []]})
methods.update({l_exp.gss_trapezoid_midpoint.__name__     : [l_exp.gss_trapezoid_midpoint, True,    10,   'y',     '--',     'x', []]})
methods.update({l_exp.scipy_pointwise.__name__ + '_vode'  : [l_exp.scipy_pointwise,        False,    7,   'r',     '--',     '.', ['vode']]})
methods.update({l_exp.scipy_pointwise.__name__ + '_lsoda' : [l_exp.scipy_pointwise,        False,    7,   'r',     '--',     '.', ['lsoda']]})
