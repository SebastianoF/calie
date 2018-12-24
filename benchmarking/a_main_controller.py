from VECtorsToolkit.operations import lie_exp

l_exp = lie_exp.LieExp()
l_exp.s_i_o = 3

# name, active [true/false]

methods = [[l_exp.scaling_and_squaring,   True],
           [l_exp.gss_ei,                 True],
           [l_exp.gss_ei_mod,             True],
           [l_exp.gss_aei,                True],
           [l_exp.midpoint,               True],
           [l_exp.series,                 True],
           [l_exp.series_mod,             True],
           [l_exp.euler,                  True],
           [l_exp.euler_aei,              True],
           [l_exp.euler_mod,              True],
           [l_exp.heun,                   True],
           [l_exp.heun_mod,               True],
           [l_exp.rk4,                    True],
           [l_exp.gss_rk4,                True],
           [l_exp.trapeziod_euler,        True],
           [l_exp.trapezoid_midpoint,     True],
           [l_exp.gss_trapezoid_euler,    True],
           [l_exp.gss_trapezoid_midpoint, True],
           [l_exp.scipy_pointwise,        True]]

# name   :   [num_steps (None is automatic) colour  line-style   marker, [sub-options]]

methods_dict = {'scaling_and_squaring':   [True,    7,    'b',      '-',     '+', []],
                'gss_ei':                 [True,    7,    'b',     '--',     '+', []],
                'gss_ei_mod':             [True,    7,    'r',      '-',     '.', []],
                'gss_aei':                [True,    7,    'r',     '--',     'x', []],
                'midpoint':               [False,   10,   'b',      '-',     '*', []],
                'series':                 [True,    10,   'c',      '-',     '.', []],
                'series_mod':             [True,    40,   'g',      '-',     '>', []],
                'euler':                  [True,    40,   'm',     '--',     '>', []],
                'euler_aei':              [True,    40,   'm',      '-',     '>', []],
                'euler_mod':              [True,    40,   'm',     '--',     '>', []],
                'heun':                   [True,    10,   'k',      '-',     '.', []],
                'heun_mod':               [True,    10,   'k',     '--',     '.', []],
                'rk4':                    [True,    10,   'y',     '--',     'x', []],
                'gss_rk4':                [True,    10,   'y',     '--',     'x', []],
                'trapezoid_euler':        [True,    10,   'y',     '--',     'x', []],
                'trapezoid_midpoint':     [True,    10,   'y',     '--',     'x', []],
                'gss_trapezoid_euler':    [True,    10,   'y',     '--',     'x', []],
                'gss_trapezoid_midpoint': [True,    10,   'y',     '--',     'x', []],
                'scipy_pointwise':        [False,    7,   'r',     '--',     '.', ['vode', 'lsoda']],
                }
