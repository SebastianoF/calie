import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def interpn(*args, **kw):
    """Interpolation on N-D.

    ai = interpn(x, y, z, ..., a, xi, yi, zi, ...)
    where the arrays x, y, z, ... define a rectangular grid
    and a.shape == (len(x), len(y), len(z), ...)
    """
    method = kw.pop('method', 'cubic')
    if kw:
        raise ValueError("Unknown arguments: " % kw.keys())
    nd = (len(args)-1)//2
    if len(args) != 2*nd+1:
        raise ValueError("Wrong number of arguments")
    q = args[:nd]
    qi = args[nd+1:]
    a = args[nd]
    for j in range(nd):
        a = interp1d(q[j], a, axis=j, kind=method)(qi[j])
    return a


x = np.linspace(0, 1, 6)
y = np.linspace(0, 1, 7)
k = np.array([0, 1])
z = np.cos(2*x[:,None,None] + k[None,None,:]) * np.sin(3*y[None,:,None])

xi = np.linspace(0, 1, 60)
yi = np.linspace(0, 1, 70)
zi = interpn(x, y, z, xi, yi, method='linear')

plt.subplot(221)
plt.imshow(z[:,:,0].T, interpolation='nearest')

plt.subplot(222)
plt.imshow(zi[:,:,0].T, interpolation='nearest')

plt.subplot(223)
plt.imshow(z[:,:,1].T, interpolation='nearest')

plt.subplot(224)
plt.imshow(zi[:,:,1].T, interpolation='nearest')

plt.show()