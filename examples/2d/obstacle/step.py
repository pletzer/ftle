import numpy as np
import matplotlib.pyplot as plt

nu = 1001
nv = 1001

d = 1 # separation
def z_transf(w):
    sqrt_w_1 = np.sqrt(w - 1)
    coeff = 2 * d * 1j / np.pi
    return coeff * (sqrt_w_1 - np.log(1 + 1j*sqrt_w_1)/(1 - 1j*sqrt_w_1)/(2*1j))

# check a few points
print(f'w = 1 => z = {z_transf(1.0 + 0j)}')
print(f'w = -1 => z = {z_transf(-1.0 + 0j)}')

umin = 0.0
umax = 1.0
vmax = 1.0
eps = 1.e-8
u = np.linspace(umin, umax, nu)
v = np.linspace(eps, vmax, nv)
uu, vv = np.meshgrid(u, v)
ww = uu + 1j*vv
# potential
potential = (np.log(ww)/np.pi).imag
#potential = np.angle(ww)/np.pi
#potential = uu
plt.pcolor(uu, vv, potential)
plt.show()

# now show the solution in z plane
zz = z_transf(ww)
xx = zz.real
yy = zz.imag
plt.contourf(xx, yy, potential)
plt.show()

# eps = 0.01
# w_radius = np.linspace(eps, 2 - eps, ni)
# w_phase = np.linspace(0, np.pi, nj)

# ww_radius, ww_phase = np.meshgrid(w_radius, w_phase)
# ww = ww_radius * np.exp(1j* ww_phase)
# zz = z_transf(ww)
# for j in range(zz.shape[0]):
#     plt.plot(zz[j,:].real, zz[j,:].imag)
# for i in range(zz.shape[1]):
#     plt.plot(zz[:, i].real, zz[:, i].imag)
# plt.axis('equal')
# # plot the geometry
# xmax = zz.real.max()
# plt.plot([0., xmax], [0., 0.], 'k-')
# ymax = zz.imag.max()
# plt.plot([0., 0.], [0., ymax], 'k-')
# xmin = zz.real.min()
# plt.plot([xmin, xmax], [-d, -d], 'k-')
plt.show()
