import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy

import defopt
from pathlib import Path

def writeVTI(data, dx, dy, dz, varname='ftle', filename: str='ftle_1200.vi'):
    """
    Write data to a VTI file
    :param filename: name of the VTI file
    """
    import pyvista as pv

    # Create a VTK grid
    grid = pv.ImageData(dimensions=data.shape)

    # Set spacing
    grid.spacing = (dz, dy, dx) 

    # Attach data (must be flattened to 1D)
    grid["ftle"] = data.flatten(order="F")
    
    # Save to a .vtk file
    grid.save(filename)


import numpy as np
from scipy.integrate import solve_ivp

def compute_ftle(ds, time_index, T, imin, imax, jmin, jmax, method='RK45', atol=1.e-8, rtol=1.e-8):
    """
    Compute FTLE (finite-time Lyapunov exponent) for a 3D grid of initial points.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain ds.u_xy, ds.v_xy, ds.w_xy at face-centered locations described in your docstring.
        First axis of each variable is time, then nz, ny, nx
    time_index : int
        Time index to take the (time-constant) velocity field from.
    T : float
        Integration time (can be positive or negative). Must be non-zero.
    imin : int
        Min index in x direction
    imax : int
        Max index in x direction, imax > imin + 1 or negetive
    jmin : int
        Min index in y direction
    jmax : int
        Max index in y direction, jmax > jmin + 1 or negative
    method : str
        Integration method for scipy.solve_ivp (e.g. 'RK45','RK23','DOP853','Radau','BDF','LSODA').
    atol, rtol : float
        Integrator tolerances.

    Returns
    -------
    ftle : ndarray
        Array with shape (nx, ny, nz) containing FTLE values.
    """

    if T == 0:
        raise ValueError("Integration time T must be non-zero.")

    # nodal grid coords (1D) for the patch we're computing the FTLE for
    x = np.asarray(ds.xu[imin: imax])
    y = np.asarray(ds.yv[jmin: jmax])
    z = np.asarray(ds.zw_xy)

    xmin, ymin, zmin = x[0], y[0], z[0]
    xmax, ymax, zmax = x[-1], y[-1], z[-1]
    print(f'xmin, xmax = {xmin}, {xmax}')
    print(f'ymin, ymax = {ymin}, {ymax}')
    print(f'zmin, zmax = {zmin}, {zmax}')
    dx, dy, dz = float(x[1] - x[0]), float(y[1] - y[0]), float(z[1] - z[0])
    print(f'dx, dy, dz = {dx}, {dy}, {dz}')
    nx, ny, nz = len(x), len(y), len(z)
    print(f'nx, ny, nz = {nx}, {ny}, {nz}')

    # create meshgrid with indexing='xy' so shape is (nz, ny, nx)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='xy')  # shapes: (nz, ny, nx)

    # flatten initial coordinates into 1D arrays length n = nx * ny * nz
    n = nx * ny * nz
    xflat = xx.ravel()
    yflat = yy.ravel()
    zflat = zz.ravel()

    # read velocity faces at the requested time index (make them numpy arrays)
    # replace Nans with zeros
    uface = np.asarray(ds.u_xy[time_index, :, jmin:jmax, imin:imax].fillna(0.0))
    vface = np.asarray(ds.v_xy[time_index, :, jmin:jmax, imin:imax].fillna(0.0))
    wface = np.asarray(ds.w_xy[time_index, :, jmin:jmax, imin:imax].fillna(0.0))

    # define RHS: returns flat vector of length 3*n
    def vel_fun(t, pos):
        # pos is 3*n vector: [x0...x_{n-1}, y0..., z0...]
        xi = pos[0:n].copy()
        yi = pos[n:2*n].copy()
        zi = pos[2*n:3*n].copy()

        # fractional cell coordinates (clamp to domain boundaries)
        kfloat = np.clip((zi - zmin) / dz, 0.0, nz - 1.0)
        jfloat = np.clip((yi - ymin) / dy, 0.0, ny - 1.0)
        ifloat = np.clip((xi - xmin) / dx, 0.0, nx - 1.0)

        # integer cell index (lower corner)
        k0 = np.clip(np.floor(kfloat).astype(int), 0, nz - 2)
        j0 = np.clip(np.floor(jfloat).astype(int), 0, ny - 2)
        i0 = np.clip(np.floor(ifloat).astype(int), 0, nx - 2)

        k1 = k0 + 1
        j1 = j0 + 1
        i1 = i0 + 1

        # local coords in cell (0..1)
        zet = kfloat - k0
        eta = jfloat - j0
        xsi = ifloat - i0
        tez = 1.0 - zet
        ate = 1.0 - eta
        isx = 1.0 - xsi

        # Interpolate based on your face-centered assumptions:
        # u is constant in y,z and linear in x between i0,i1 (shape: (nz-1, ny-1, nx))
        # v is linear in y, etc...
        # Keep same indexing order as your data arrays — if ds stores arrays with different axis order
        # you must reorder accordingly. Below I assume uface[k,j,i], consistent with your code.

        ui = uface[k0, j0, i0] * isx + uface[k0, j0, i1] * xsi
        vi = vface[k0, j0, i0] * ate + vface[k0, j1, i0] * eta
        wi = wface[k0, j0, i0] * tez + wface[k1, j0, i0] * zet

        return np.concatenate([ui, vi, wi])

    # initial condition: concatenated positions
    y0 = np.concatenate([xflat, yflat, zflat])  # length 3*n

    # integrate over the whole interval [0, T]
    result = solve_ivp(fun=vel_fun, t_span=(0.0, float(T)), y0=y0,
                       method=method, atol=atol, rtol=rtol)

    if not result.success:
        raise RuntimeError(f"ODE integrator failed: {result.message}")

    final_pos = result.y[:, -1]  # length 3*n

    Xf = final_pos[0:n].reshape((nx, ny, nz))
    Yf = final_pos[n:2*n].reshape((nx, ny, nz))
    Zf = final_pos[2*n:3*n].reshape((nx, ny, nz))

    # Preallocate gradient arrays (same shape)
    f11 = np.empty_like(Xf); f12 = np.empty_like(Xf); f13 = np.empty_like(Xf)
    f21 = np.empty_like(Xf); f22 = np.empty_like(Xf); f23 = np.empty_like(Xf)
    f31 = np.empty_like(Xf); f32 = np.empty_like(Xf); f33 = np.empty_like(Xf)

    # d{X,Y,Z}/dx  -- last axis is x (axis 2) because of (nx,ny,nz) ordering
    # central differences for interior
    f11[:, :, 1:-1] = (Xf[:, :, 2:] - Xf[:, :, :-2]) / (2.0 * dx)
    f21[:, :, 1:-1] = (Yf[:, :, 2:] - Yf[:, :, :-2]) / (2.0 * dx)
    f31[:, :, 1:-1] = (Zf[:, :, 2:] - Zf[:, :, :-2]) / (2.0 * dx)
    # one-sided at boundaries
    f11[:, :, 0] = (Xf[:, :, 1] - Xf[:, :, 0]) / dx
    f21[:, :, 0] = (Yf[:, :, 1] - Yf[:, :, 0]) / dx
    f31[:, :, 0] = (Zf[:, :, 1] - Zf[:, :, 0]) / dx
    f11[:, :, -1] = (Xf[:, :, -1] - Xf[:, :, -2]) / dx
    f21[:, :, -1] = (Yf[:, :, -1] - Yf[:, :, -2]) / dx
    f31[:, :, -1] = (Zf[:, :, -1] - Zf[:, :, -2]) / dx

    # d{X,Y,Z}/dy  -- middle axis is y (axis 1)
    f12[:, 1:-1, :] = (Xf[:, 2:, :] - Xf[:, :-2, :]) / (2.0 * dy)
    f22[:, 1:-1, :] = (Yf[:, 2:, :] - Yf[:, :-2, :]) / (2.0 * dy)
    f32[:, 1:-1, :] = (Zf[:, 2:, :] - Zf[:, :-2, :]) / (2.0 * dy)
    f12[:, 0, :] = (Xf[:, 1, :] - Xf[:, 0, :]) / dy
    f22[:, 0, :] = (Yf[:, 1, :] - Yf[:, 0, :]) / dy
    f32[:, 0, :] = (Zf[:, 1, :] - Zf[:, 0, :]) / dy
    f12[:, -1, :] = (Xf[:, -1, :] - Xf[:, -2, :]) / dy
    f22[:, -1, :] = (Yf[:, -1, :] - Yf[:, -2, :]) / dy
    f32[:, -1, :] = (Zf[:, -1, :] - Zf[:, -2, :]) / dy

    # d{X,Y,Z}/dz  -- first axis is z (axis 0)
    f13[1:-1, :, :] = (Xf[2:, :, :] - Xf[:-2, :, :]) / (2.0 * dz)
    f23[1:-1, :, :] = (Yf[2:, :, :] - Yf[:-2, :, :]) / (2.0 * dz)
    f33[1:-1, :, :] = (Zf[2:, :, :] - Zf[:-2, :, :]) / (2.0 * dz)
    f13[0, :, :] = (Xf[1, :, :] - Xf[0, :, :]) / dz
    f23[0, :, :] = (Yf[1, :, :] - Yf[0, :, :]) / dz
    f33[0, :, :] = (Zf[1, :, :] - Zf[0, :, :]) / dz
    f13[-1, :, :] = (Xf[-1, :, :] - Xf[-2, :, :]) / dz
    f23[-1, :, :] = (Yf[-1, :, :] - Yf[-2, :, :]) / dz
    f33[-1, :, :] = (Zf[-1, :, :] - Zf[-2, :, :]) / dz

    # Cauchy-Green tensor components
    C11 = f11*f11 + f21*f21 + f31*f31
    C12 = f11*f12 + f21*f22 + f31*f32
    C13 = f11*f13 + f21*f23 + f31*f33
    C22 = f12*f12 + f22*f22 + f32*f32
    C23 = f12*f13 + f22*f23 + f32*f33
    C33 = f13*f13 + f23*f23 + f33*f33

    # Build (nx, ny, nz, 3, 3) array and compute eigenvalues
    C = np.empty((nx, ny, nz, 3, 3), dtype=C11.dtype)
    C[..., 0, 0] = C11
    C[..., 0, 1] = C12
    C[..., 0, 2] = C13
    C[..., 1, 0] = C12
    C[..., 1, 1] = C22
    C[..., 1, 2] = C23
    C[..., 2, 0] = C13
    C[..., 2, 1] = C23
    C[..., 2, 2] = C33

    C_flat = C.reshape(-1, 3, 3)
    eigvals = np.linalg.eigvalsh(C_flat)  # sorted ascending
    max_lambda = eigvals[:, -1].reshape((nx, ny, nz))

    # avoid log domain errors
    eps = 1e-16
    max_lambda = np.clip(max_lambda, eps, None)

    ftle = np.log(max_lambda) / (2.0 * abs(float(T)))
    return ftle


def main(*, filename: str='small_blf_day_loc1_4m_xy_N04.003.nc', 
    save_dir: str='./test', 
    t_start: int=10, t_end: int=11, 
    T: int=-10, 
    imin: int=0, imax: int=-1,
    jmin: int=0, jmax: int=-1):
    """
    Compute the FTLE
    @param filename input PALM NetCDF file name
    @param save_dir directory to save the output file
    @param t_start starting time index
    @param t_end one past last time index
    @param T trajectory integration time
    @param imin min index in x direction
    @param imax max index in x direction
    @param jmin min index in y direction
    @param jmax max index in y direction
    """

    # read the data and select times
    ds = xr.open_dataset(filename, engine='netcdf4', decode_timedelta=False)

    dx = (ds.xu[1] - ds.xu[0]).item()
    dy = (ds.yv[1] - ds.yv[0]).item()
    dz = (ds.zu_xy[1] - ds.zu_xy[0]).item()

    print(f'time = {ds.time}')

    for time_index in range(t_start, t_end):

        print(f'time index = {time_index}...')
        ftle = compute_ftle(ds, time_index, T=T, imin=imin, imax=imax, jmin=jmin, jmax=jmax)

        writeVTI(ftle, dx, dy, dz, varname='ftle', filename=f'ftle_{time_index:04d}.vi')


if __name__ == '__main__':
    defopt.run(main)

