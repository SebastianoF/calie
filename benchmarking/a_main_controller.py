from VECtorsToolkit.operations import lie_exp

l_exp = lie_exp.LieExp()
l_exp.s_i_o = 3

# name, active num_steps_default (None is automatic) colour  line-style   marker, [sub-options]]


methods = [[l_exp.scaling_and_squaring,   True,    7,    'b',      '-',     '+', []],
           [l_exp.gss_ei,                 True,    7,    'b',      '-',     '+', []],
           [l_exp.gss_ei_mod,             True,    7,    'b',     '--',     '+', []],
           [l_exp.gss_aei,                True,    7,    'r',      '-',     '.', []],
           [l_exp.midpoint,               True,    7,    'r',     '--',     'x', []],
           [l_exp.series,                 True,    10,   'b',      '-',     '*', []],
           [l_exp.series_mod,             True,    10,   'c',      '-',     '.', []],
           [l_exp.euler,                  True,    40,   'g',      '-',     '>', []],
           [l_exp.euler_aei,              True,    40,   'm',     '--',     '>', []],
           [l_exp.euler_mod,              True,    40,   'm',      '-',     '>', []],
           [l_exp.heun,                   True,    40,   'm',     '--',     '>', []],
           [l_exp.heun_mod,               True,    10,   'k',      '-',     '.', []],
           [l_exp.rk4,                    True,    10,   'k',     '--',     '.', []],
           [l_exp.gss_rk4,                True,    10,   'y',     '--',     'x', []],
           [l_exp.trapeziod_euler,        True,    10,   'y',     '--',     'x', []],
           [l_exp.trapezoid_midpoint,     True,    10,   'y',     '--',     'x', []],
           [l_exp.gss_trapezoid_euler,    True,    10,   'y',     '--',     'x', []],
           [l_exp.gss_trapezoid_midpoint, True,    10,   'y',     '--',     'x', []],
           [l_exp.scipy_pointwise,        True,     7,   'r',     '--',     '.', ['vode', 'lsoda']]]
