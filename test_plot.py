import pyvista as pv
import numpy as np

nx, ny, nz = 100, 110, 120
x = np.linspace(0, 2., nx)
y = np.linspace(0, 3., ny)
z = np.linspace(0., 1., nz)
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]

zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
ff = np.cos(np.pi*xx) * np.cos(4*np.pi*yy) * zz * (z[-1] - zz)
print(f'dimensions: ff.shape={ff.shape} nx={nx} ny={ny} nz={nz}')

grid = pv.ImageData()
grid.dimensions = (nx, ny, nz)
grid.spacing = (dx, dy, dz)
#grid.origin = (x[0], y[0], z[0])
grid.point_data['ff'] = ff.flatten(order='C')
grid.plot(show_edges=False)

