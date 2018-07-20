#!/usr/bin/env python
import numpy as np
import scipy.optimize, scipy.ndimage
import matplotlib.pyplot as plt
import time

# read a reference image
lena = scipy.ndimage.imread('lena-256.png', 'F')
# lena = lena[100:150, 100:150]

# create a reference velocity field
vel_field = np.zeros([10, 20, 2])

passepartout = 10

vel_field[:, :, 0] = lena[100:110, 100:120]-np.mean(lena[100:110, 100:120].ravel())
vel_field[:, :, 1] = lena[200:210, 200:220]-np.mean(lena[200:210, 200:220].ravel())

#vel_field[4, 10, 0]=10
#vel_field[6, 15, 1]=-10

num_dims = 2
vel_field_size = vel_field.shape
im_size = vel_field.shape[0:2]
num_pix = np.prod(vel_field.shape[0:2])
num_field_elem = vel_field.size
print('num_field_elem = {}\n num_pix = {}\n im_size = {}\n'.format(num_field_elem, num_pix, im_size))

sig = 1
g_kern = np.array([1., -1.])
#g_kern = 0.5*np.array([1., 0, -1.])
d_kern = np.array([0, 1., -1.])
#d_kern = g_kern

# define gradient operators
GradXOp = scipy.sparse.linalg.LinearOperator(
        [num_pix, num_pix], 
        matvec = lambda y: scipy.ndimage.convolve1d( y.reshape(im_size), weights=g_kern, axis=0, mode='constant', cval=0.0).ravel()
        #matvec = lambda y: np.gradient(y.reshape(im_size), axis=0).ravel()
        #matvec = lambda y: scipy.ndimage.filters.gaussian_filter1d(y.reshape(im_size), sigma=sig, order=1, axis=0, mode='constant', cval=0.0).ravel()
        )

GradYOp = scipy.sparse.linalg.LinearOperator(
        [num_pix, num_pix], 
        matvec = lambda y: scipy.ndimage.convolve1d( y.reshape(im_size), weights=g_kern, axis=1, mode='constant', cval=0.0 ).ravel()
        #matvec = lambda y: np.gradient(y.reshape(im_size), axis=1).ravel()
        #matvec = lambda y: scipy.ndimage.filters.gaussian_filter1d(y.reshape(im_size), sigma=sig, order=1, axis=1, mode='constant', cval=0.0).ravel()
        )

GradOp = scipy.sparse.linalg.LinearOperator(
        [2*num_pix, num_pix], 
        matvec = lambda y: np.dstack([GradXOp.matvec(y).reshape(im_size), GradYOp.matvec(y).reshape(im_size)]).ravel()
        )

# define the divergence operator
DivergenceOp = scipy.sparse.linalg.LinearOperator(
        [num_pix, num_field_elem], 
        #matvec = lambda y: np.ufunc.reduce(np.add, [np.gradient(y.reshape(vel_field_size)[:, :, i], axis=i) for i in range(num_dims)]).ravel()
        #matvec = lambda y: np.ufunc.reduce(np.add, [GradXOp.matvec(y.reshape(vel_field_size)[:, :, 0].ravel()), GradYOp.matvec(y.reshape(vel_field_size)[:, :, 1].ravel())])
        matvec = lambda y: scipy.ndimage.convolve1d( y.reshape(vel_field_size)[:, :, 0], weights=d_kern, axis=0, mode='constant', cval=0.0 ).ravel() + 
                           scipy.ndimage.convolve1d( y.reshape(vel_field_size)[:, :, 1], weights=d_kern, axis=1, mode='constant', cval=0.0 ).ravel()
        )

# define the Lapace operator
LaplaceOp = scipy.sparse.linalg.LinearOperator(
        [num_pix, num_pix], 
        #matvec = lambda y: scipy.ndimage.laplace( y.reshape(im_size), mode='constant', cval=0.0 ).ravel(), 
        #rmatvec = lambda y: scipy.ndimage.laplace( y.reshape(im_size), mode='constant', cval=0.0 ).ravel()
        matvec = lambda y: DivergenceOp.matvec(GradOp.matvec(y)), 
        rmatvec = lambda y: DivergenceOp.matvec(GradOp.matvec(y))
        #matvec = lambda y: scipy.ndimage.filters.gaussian_laplace( y.reshape(im_size), sigma=sig, mode='constant', cval=0.0 ).ravel()
        )

# compute the divergence of the input field
vel_field_div = DivergenceOp.matvec(vel_field.ravel()).reshape(im_size)

# apply the laplace operator to the reference image
#lena_laplace = LaplaceOp.matvec(lena.ravel())

# solve the inverse problem
start = time.time()
lres_pack = scipy.sparse.linalg.cg(LaplaceOp, vel_field_div.ravel())
end = time.time()
pressure = lres_pack[0].reshape(im_size)

# check the result numerically
print(' ')
print('cg - ', end-start, 'sec')
print('min = ', np.min(pressure.ravel()))
print('max = ', np.max(pressure.ravel()))
print('residuals = ', np.linalg.norm(LaplaceOp.matvec(pressure.ravel())-vel_field_div.ravel()))

#grad_p =
#print grad_p.shape
grad_p = GradOp.matvec(pressure.ravel()).reshape(vel_field_size)
#grad_p = np.zeros(vel_field.shape)
#grad_p[:, :, 0] = GradXOp.matvec(pressure.ravel()).reshape(im_size)
#grad_p[:, :, 1] = GradYOp.matvec(pressure.ravel()).reshape(im_size)

#grad_p = np.zeros(vel_field.shape)
#[grad_p[:, :, 0], grad_p[:, :, 1]] = np.asarray(np.gradient(pressure))

# compute the divergence
div_grad_p = DivergenceOp.matvec(grad_p.ravel()).reshape(im_size)
#div_grad_pb = DivergenceOp.matvec(grad_pb.ravel()).reshape(im_size)

# check the result numerically
print(' ')
print('div grad')
print('residuals = ', np.linalg.norm(vel_field_div.ravel()-div_grad_p.ravel()))
#print 'residualsb = ', np.linalg.norm(vel_field_div.ravel()-div_grad_pb.ravel())

# compute the divergence-free field
divfree_field = vel_field - grad_p;

div_divfree_field = DivergenceOp.matvec(divfree_field.ravel()).reshape(im_size)

div_vel_field = DivergenceOp.matvec(vel_field.ravel()).reshape(im_size)

# check the result numerically
print(' ')
print('divfree')
print('init div norm = ', np.linalg.norm(div_vel_field.ravel()))
print('new div norm = ', np.linalg.norm(div_divfree_field.ravel()))

# display the images
plt.figure(1)

ax = plt.subplot(3, 3, 1)
ax.set_title('u_x')
plt.imshow(vel_field[:, :, 0], cmap='gray')

ax = plt.subplot(3, 3, 2)
ax.set_title('u_y')
plt.imshow(vel_field[:, :, 1], cmap='gray')

ax = plt.subplot(3, 3, 4)
ax.set_title('div_u')
plt.imshow(vel_field_div, cmap='gray')

ax = plt.subplot(3, 3, 7)
ax.set_title('p')
plt.imshow(pressure, cmap='gray')

ax = plt.subplot(3, 3, 8)
ax.set_title('grad_p_x')
plt.imshow(grad_p[:, :, 0], cmap='gray')

ax = plt.subplot(3, 3, 9)
ax.set_title('grad_p_y')
plt.imshow(grad_p[:, :, 1], cmap='gray')

ax = plt.subplot(3, 3, 5)
ax.set_title('div_grad_p')
plt.imshow(div_grad_p, cmap='gray')

#ax = plt.subplot(3, 3, 6)
#ax.set_title('div_grad_pb')
#plt.imshow(div_grad_pb, cmap='gray')

plt.figure(2)

ax = plt.subplot(2, 3, 1)
ax.set_title('div_grad_p')
plt.imshow(div_grad_p, cmap='gray')

ax = plt.subplot(2, 3, 2)
ax.set_title('laplacian_p')
plt.imshow(LaplaceOp.matvec(pressure.ravel()).reshape(im_size), cmap='gray')

ax = plt.subplot(2, 3, 3)
ax.set_title('div_u')
plt.imshow(div_vel_field, cmap='gray')
plt.colorbar()

ax = plt.subplot(2, 3, 4)
ax.set_title('dv_x')
plt.imshow(divfree_field[:, :, 0], cmap='gray')
plt.colorbar()

ax = plt.subplot(2, 3, 5)
ax.set_title('dv_y')
plt.imshow(divfree_field[:, :, 1], cmap='gray')
plt.colorbar()

ax = plt.subplot(2, 3, 6)
ax.set_title('div_dv')
plt.imshow(div_divfree_field, cmap='gray')
plt.colorbar()



plt.figure(3)

ax = plt.subplot(2, 2, 1)
ax.set_title('u_x')
plt.imshow(vel_field[:, :, 0], cmap='gray')

lxa = LaplaceOp.matvec(vel_field[:, :, 0].ravel()).reshape(im_size)
lxb = scipy.ndimage.laplace( vel_field[:, :, 0], mode='constant', cval=0.0 )

ax = plt.subplot(2, 2, 3)
ax.set_title('lxa')
plt.imshow(lxa, cmap='gray')
plt.colorbar()

ax = plt.subplot(2, 2, 4)
ax.set_title('lxb')
plt.imshow(lxb, cmap='gray')
plt.colorbar()


plt.show()
