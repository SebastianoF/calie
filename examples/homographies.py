import numpy as np
import matplotlib.pyplot as plt
from sympy.core.cache import clear_cache

from calie.transformations import pgl2
from calie.operations import lie_exp
from calie.visualisations.fields import fields_at_the_window
from calie.fields import generate as gen
from calie.fields import queries as qr


if __name__ == "__main__":

    clear_cache()

    random_seed = 5

    if random_seed > 0:
        np.random.seed(random_seed)

    s_i_o = 3
    pp = 2

    # Parameters SVF:
    x_1, y_1, z_1 = 50, 50, 1

    if z_1 == 1:
        d = 2
        domain = (x_1, y_1)
        shape = list(domain) + [1, 1, 2]

        # center of the homography
        x_c = x_1 / 2
        y_c = y_1 / 2
        z_c = 1

        projective_center = [x_c, y_c, z_c]

    else:
        d = 3
        domain = (x_1, y_1, z_1)
        shape = list(domain) + [1, 3]

        # center of the homography
        x_c = x_1 / 2
        y_c = y_1 / 2
        z_c = z_1 / 2
        w_c = 1

        projective_center = [x_c, y_c, z_c, w_c]

    print('---------------------')
    print('Computations started!')
    print('---------------------')

    scale_factor = 1. / (np.max(domain) * 3)
    special = False
    sigma = 1
    hom_attributes = [d, scale_factor, sigma, special]

    h_a, h_g = pgl2.get_random_hom_matrices(d=hom_attributes[0],
                                            scale_factor=hom_attributes[1],
                                            sigma=hom_attributes[2],
                                            special=hom_attributes[3],
                                            projective_center=np.array([25, 25]))

    print(h_a)
    print(h_g)

    svf1 = gen.generate_from_projective_matrix(domain, h_a, structure='algebra')
    flow = gen.generate_from_projective_matrix(domain, h_g, structure='group')

    l_exp = lie_exp.LieExp()
    flow_ss = l_exp.scaling_and_squaring(svf1, input_num_steps=10)

    print(qr.norm(flow - flow_ss, passe_partout_size=4))

    fields_at_the_window.see_field(svf1, input_color='r')
    fields_at_the_window.see_field(flow, input_color='b')
    fields_at_the_window.see_field(flow_ss, input_color='g', width=0.01)

    plt.show()

    for tp in [1, 2, 3, 4, 6, 7, 10, 12, 20]:
        flow_ss = l_exp.scaling_and_squaring(svf1, input_num_steps=tp)
        print(qr.norm(flow - flow_ss, passe_partout_size=4))
