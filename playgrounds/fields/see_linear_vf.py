from matplotlib import pyplot
import scipy

from calie.operations import lie_exp
from calie.visualisations.fields.fields_at_the_window import see_field
from calie.visualisations.fields.fields_comparisons import see_n_fields_separate
from calie.fields import generate as gen
from calie.transformations import linear
from calie.fields import queries as qr

if __name__ == '__main__':

    pyplot.close('all')
    l_exp = lie_exp.LieExp()

    # ----------------------------
    # ---- First example ---------
    # ----------------------------

    taste = 3
    alpha = 1
    beta = 0.1

    dm = beta * linear.randomgen_linear_by_taste(1, taste, (20, 20))
    svf = alpha * gen.generate_from_matrix((40, 40), dm, structure='algebra')

    # get integral of the SVF with generalised scaling and squaring.
    sdisp = l_exp.gss_aei(svf)
    see_field(svf, subtract_id=False, input_color='r', fig_tag=1)
    see_field(sdisp, subtract_id=False, input_color='b', title_input='Unstable node exp', fig_tag=1)

    pyplot.show(block=False)

    # get integral of the SVF with the matrix exponential
    m = scipy.linalg.expm(dm)
    sdisp_expm = alpha * gen.generate_from_matrix((40, 40), m, structure='group')

    see_field(svf, subtract_id=False, input_color='r', fig_tag=2)
    see_field(sdisp_expm, subtract_id=False, input_color='b', title_input='Unstable node expm', fig_tag=2)

    pyplot.show(block=False)

    print('difference in "improper" norm of the flow fields {}'.format(
        qr.norm(sdisp - sdisp_expm, passe_partout_size=5))
    )

    # ----------------------------
    # ---- See all tastes --------
    # ----------------------------
    vfs = []
    titles = []
    for taste in range(1, 7):
        dm = linear.randomgen_linear_by_taste(1, taste, (20, 20))
        svf = beta * gen.generate_from_matrix((40, 40), dm, structure='algebra')
        vfs.append(svf)
        titles.append('Kind {}'.format(taste))

    extra_titles = {'Kind 1': 'Unstable node',
                    'Kind 2': 'Stable node',
                    'Kind 3': 'Saddle',
                    'Kind 4': 'Outward spiral',
                    'Kind 5': 'Inward spiral',
                    'Kind 6': 'circles'}

    titles = [ti + ': ' + extra_titles[ti] for ti in titles]

    see_n_fields_separate(vfs,
                          row_fig=2,
                          col_fig=3,
                          input_color=['r'] * 6,
                          input_figsize=(15, 8),
                          title_input=titles,
                          fig_tag=3,
                          xy_lims=(15, 25, 15, 25),
                          width=0.05)
    pyplot.show(block=True)
