import os
from matplotlib import pylab


def snapshot_vf(t, pos, vel, colors, sigma, arrow_scale=.2, output_dir=None, frame_num=0):
    """ Create single frame for vector field """
    pylab.cla()
    pylab.axis([0, 1, 0, 1])
    pylab.setp(pylab.gca(), xticks=[0, 1], yticks=[0, 1])
    for (x, y), (dx, dy), c in zip(pos, vel, colors):
        dx *= arrow_scale
        dy *= arrow_scale
        circle = pylab.Circle((x, y), radius=sigma, fc=c)
        pylab.gca().add_patch(circle)
        pylab.arrow(x, y, dx, dy, fc="k", ec="k", head_width=0.05, head_length=0.05)
    pylab.text(.5, 1.03, 't = %.2f' % t, ha='center')
    pylab.savefig(os.path.join(output_dir, '%04i.png' % frame_num))
    frame_num += 1


def snapshot_ic():
    """ Create a single frame for integral curve, given a static vector field """
    pass


def snapshot_sf():
    """ Create a single frame for a scalar field """
    pass


def get_evolution(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
