import numpy as np
from matplotlib import pyplot

from VECtorsToolkit.operations import lie_exp
from VECtorsToolkit.visualisations.fields.fields_at_the_window import see_field
from VECtorsToolkit.visualisations.fields.fields_comparisons import see_n_fields_separate
from VECtorsToolkit.fields import generate as gen
from VECtorsToolkit.transformations import linear

if __name__ == '__main__':

    pyplot.close('all')

    # ----------------------------
    # ---- First example ---------
    # ----------------------------

    l_exp = lie_exp.LieExp()
    taste = 1

    dm = linear.randomgen_linear_by_taste(1, 1, (20, 20))
    svf = 0.1 * gen.generate_from_matrix((40, 40), dm, structure='algebra')

    sdisp = l_exp.gss_aei(svf)
    see_field(1 * svf, subtract_id=False, input_color='r', fig_tag=1)
    see_field(sdisp, subtract_id=False, input_color='b', title_input='Unstable node', fig_tag=1)

    pyplot.show(block=False)

    # ----------------------------
    # ---- See all tastes --------
    # ----------------------------
    vfs = []
    titles = []
    for taste in range(1, 7):
        dm = linear.randomgen_linear_by_taste(1, taste, (20, 20))
        svf = 0.1 * gen.generate_from_matrix((40, 40), dm, structure='algebra')
        vfs.append(svf)
        titles.append('Taste {}'.format(taste))

    extra_titles = {'Taste 1': 'Unstable node',
                    'Taste 2': 'Stable node',
                    'Taste 3': 'Saddle',
                    'Taste 4': 'Outward spiral',
                    'Taste 5': 'Inward spiral',
                    'Taste 6': 'circles'}

    titles = [ti + ': ' + extra_titles[ti] for ti in titles]

    see_n_fields_separate(vfs,
                          row_fig=2,
                          col_fig=3,
                          input_color=['r'] * 6,
                          input_figsize=(15, 8),
                          title_input=titles,
                          fig_tag=2,
                          xy_lims=(10, 30, 10, 30))
    pyplot.show(block=False)
