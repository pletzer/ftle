import numpy as np
import matplotlib.pyplot as plt

ni = 100
nj = 8

d = 1 # separation
def z_transf(w):
    sqrt_w_1 = np.sqrt(w - 1)
    coeff = 2 * d * 1j / np.pi
    return coeff * (sqrt_w_1 - np.log(1 + 1j*sqrt_w_1)/(1 - 1j*sqrt_w_1)/(2*1j))

eps = 0.0001
w_radius = np.linspace(eps, 2.0, ni)
w_phase = np.linspace(eps, np.pi-eps, nj)

ww_radius, ww_phase = np.meshgrid(w_radius, w_phase)
ww = ww_radius * np.exp(1j* ww_phase)
zz = z_transf(ww)
for j in range(zz.shape[0]):
    plt.plot(zz[j,:].real, zz[j,:].imag)
for i in range(zz.shape[1]):
    plt.plot(zz[:, i].real, zz[:, i].imag)
plt.axis('equal')
# plot the geometry
xmax = zz.real.max()
plt.plot([0., xmax], [0., 0.], 'k-')
ymax = zz.imag.max()
plt.plot([0., 0.], [0., ymax], 'k-')
xmin = zz.real.min()
plt.plot([xmin, xmax], [-d, -d], 'k-')
plt.show()
