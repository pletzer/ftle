import xarray as xr
import numpy as np
import defopt
import pyvista as pv
from numba import njit


def compute_buildings(ds):
    """
    Find the buildings in the domain

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain ds.u_xy, ds.v_xy, ds.w_xy at face-centered locations described in your docstring.
        First axis of each variable is time, then nz, ny, nx

    Returns
    -------
    building : ndarray
        Array with shape (nx, ny, nz) containing 1 for buildings and 0 elsewhere
    """


    # nodal grid coords (1D) for the patch we're computing the FTLE for
    x = np.asarray(ds.xu)
    y = np.asarray(ds.yv)
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
    print(f'shape of velocities: {ds.u_xy.shape}')

    zz, yy, xx =  np.meshgrid(z, y, x, indexing='ij') # shapes: (nz, ny, nx)
    print(f'shape of xx: {xx.shape}')
    assert xx.shape[0] == nz
    assert xx.shape[1] == ny
    assert xx.shape[2] == nx



    # flatten initial coordinates into 1D arrays length n = nx * ny * nz
    n = nx * ny * nz
    xflat = xx.ravel()
    yflat = yy.ravel()
    zflat = zz.ravel()

    # read vertical velocities on faces at 0 time index (make them numpy arrays)
    # replace Nans with zeros
    wface = np.asarray(ds.w_xy[0, ...])
    buildings = np.array(np.isnan(wface), dtype=int)
    return buildings



def main(*, filename: str='small_blf_day_loc1_4m_xy_N04.003.nc'):
    """
    Compute the FTLE
    @param filename input PALM NetCDF file name
    """

    # read the data and select times
    ds = xr.open_dataset(filename, engine='netcdf4', decode_timedelta=False)

    buildings = compute_buildings(ds)

    dx = (ds.xu[1] - ds.xu[0]).item()
    dy = (ds.yv[1] - ds.yv[0]).item()
    dz = (ds.zw_xy[1] - ds.zw_xy[0]).item()


    nz, ny, nx = buildings.shape

    # write the data to file
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (dx, dy, dz)
    grid.point_data['buildings'] = buildings.flatten(order='C')
    grid.save(f'buildings.vti')
    #grid.plot(show_edges=False)

if __name__ == '__main__':
    defopt.run(main)

