import xarray as xr
import numpy as np
from math import copysign
import glob
import matplotlib.pyplot as plt
import scipy

from flows_3D import get_interp_arrays_3D, get_flow_3D
from integration_3D import flowmap_grid_3D
#from diagnostics_3D_gpu import ftle_grid_3D
from diagnostics_3D import ftle_grid_3D

import defopt
from pathlib import Path

"""
Instructions:

Load conda environment
    $ cd /home/tpl29
    $ source conda_env.sh

Navigate to directory with 3D FTLE code
$cd /nesi/nobackup/uc03250/dli88/palm_out/JOBS/test_files_Tam

Run script:
$python 3dFTLE_workflow.py 

After, convert all *.vti to pvd file:
filenames = sorted(glob.glob('ftle3D_*.vti'))
entries = [(int(fname.split('_')[1].split('.')[0]), fname) for fname in filenames]
write_pvd_file(entries, '.', base_filename="ftle3D")

"""
def writeFTLEVTI(var_full, filename: str='ftle_1200.vi'):
    """
    Write data to a VTI file
    :param filename: name of the VTI file
    """
    import pyvista as pv

    # Create some 3D data (example: 10x10x10 grid)
    nx, ny, nz = var_full.shape

    # Create a VTK grid
    grid = pv.ImageData(dimensions=(nx, ny, nz))

    # Set spacing
    grid.spacing = (4, 4, 4)  

    # Attach data (must be flattened to 1D)
    grid["ftle3D"] = var_full.flatten(order="F")
    
    # Save to a .vtk file
    grid.save(filename)


def compute_ftle(ds, time_index, T, nsteps=1, method='RK45', atol=1.e-8, rtol=1.e-8):

    x = ds.x
    y = ds.y
    z = ds.zu_xy
    xmin, ymin, zmin = x[0], y[0], z[0]
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    nx, ny, nz = len(x), len(y), len(z)
    xx, yy, zz = np.meshgrid(x, y, z)
    xflat = xx.reshape(-1)
    yflat = yy.reshape(-1)
    zflat = zz.reshape(-1)
    n = len(xflat)

    # read the data from file
    uface = ds.u[time_index, ...]
    vface = ds.v[time_index, ...]
    wface = ds.w[time_index, ...]

    # integrate the trajectories (flow)
    def vel_fun(t, pos):

        # get the x, y, z coordinates
        xi, yi, zi = pos[0:n].copy(), pos[n:2*n].copy(), pos[2*n:3*n]

        # find the cell
        kfloat = np.clip( (zi - zmin) / dz, 0, nz - 1)
        jfloat = np.clip( (yi - ymin) / dy, 0, ny - 1)
        ifloat = np.clip( (xi - xmin) / dx, 0, nx - 1)
        k0 = np.clip( np.floor(kfloat).astype(int), 0, nz - 2)
        j0 = np.clip( np.floor(jfloat).astype(int), 0, ny - 2)
        i0 = np.clip( np.floor(ifloat).astype(int), 0, nx - 2)
        k1 = k0 + 1
        j1 = j0 + 1
        i1 = i0 + 1

        # compute the parametric coordinates of the cell
        zet = kfloat - k0
        eta = jfloat - j0
        xsi = ifloat - i0
        tez = 1. - zet
        ate = 1. - eta
        isx = 1. - xsi
        
        ui = uface[k0, j0, i0]*isx + uface[k0, j0, i1]*xsi
        vi = vface[k0, j0, i0]*ate + vface[k0, j1, i0]*eta
        wi = wface[k0, j0, i0]*tez + wface[k1, j0, i0]*zet

        return np.concatenate([ui, vi, wi])


    dt = T / nsteps
    xyz = np.concatenate([xflat, yflat, zflat]) # flat array of initial positions
    for istep in range(nsteps):
        result = scipy.integrate.solve_ivp(vel_fun,
                            t_span=[istep*dt, (istep + 1)*dt],
                            y0=xyz, # initial condition
                            method=method,
                            atol=atol, rtol=rtol)
    
    xyz = result.y[:, -1] # get last position
    Xf = xyz[0*n:1*n].reshape(xx.shape)
    Yf = xyz[1*n:2*n].reshape(xx.shape)
    Zf = xyz[2*n:3*n].reshape(xx.shape)

    f11 = np.empty_like(Xf)
    f12 = np.empty_like(Xf)
    f13 = np.empty_like(Xf)
    f21 = np.empty_like(Xf)
    f22 = np.empty_like(Xf)
    f23 = np.empty_like(Xf)
    f31 = np.empty_like(Xf)
    f32 = np.empty_like(Xf)
    f33 = np.empty_like(Xf)

    #
    # compute the deformation gradient
    #

    # d{X, Y, Z}/dx
    f11[:, :, 1:-1] = (Xf[:, :, 2:] - Xf[:, :, :-2]) / (2*dx)
    f21[:, :, 1:-1] = (Yf[:, :, 2:] - Yf[:, :, :-2]) / (2*dx)
    f31[:, :, 1:-1] = (Zf[:, :, 2:] - Zf[:, :, :-2]) / (2*dx)

    f11[:, :, 0] = (Xf[:, :, 1] - Xf[:, :, 0]) / dx # one sided difference
    f21[:, :, 0] = (Yf[:, :, 1] - Yf[:, :, 0]) / dx # one sided difference
    f31[:, :, 0] = (Zf[:, :, 1] - Zf[:, :, 0]) / dx # one sided difference

    f11[:, :, -1] = (Xf[:, :, -1] - Xf[:, :, -2]) / dx # one sided difference
    f21[:, :, -1] = (Yf[:, :, -1] - Yf[:, :, -2]) / dx # one sided difference
    f31[:, :, -1] = (Zf[:, :, -1] - Zf[:, :, -2]) / dx # one sided difference

    # d{X, Y, Z}/dy
    f12[:, 1:-1, :] = (Xf[:, 2:, :] - Xf[:, :-2, :]) / (2*dy)
    f22[:, 1:-1, :] = (Yf[:, 2:, :] - Yf[:, :-2, :]) / (2*dy)
    f32[:, 1:-1, :] = (Zf[:, 2:, :] - Zf[:, :-2, :]) / (2*dy)

    f12[:, 0, :] = (Xf[:, 1, :] - Xf[:, 0, :]) / dy # one sided difference
    f22[:, 0, :] = (Yf[:, 1, :] - Yf[:, 0, :]) / dy # one sided difference
    f32[:, 0, :] = (Zf[:, 1, :] - Zf[:, 0, :]) / dy # one sided difference

    f12[:, -1, :] = (Xf[:, -1, :] - Xf[:, -2, :]) / dy # one sided difference
    f22[:, -1, :] = (Yf[:, -1, :] - Yf[:, -2, :]) / dy # one sided difference
    f32[:, -1, :] = (Zf[:, -1, :] - Zf[:, -2, :]) / dy # one sided difference

    # d{X, Y, Z}/dz
    f13[1:-1, :, :] = (Xf[2:, :, :] - Xf[:-2, :, :]) / (2*dz)
    f23[1:-1, :, :] = (Yf[2:, :, :] - Yf[:-2, :, :]) / (2*dz)
    f33[1:-1, :, :] = (Zf[2:, :, :] - Zf[:-2, :, :]) / (2*dz)

    f13[0, :, :] = (Xf[1, :, :] - Xf[0, :, :]) / dz # one sided difference
    f23[0, :, :] = (Yf[1, :, :] - Yf[0, :, :]) / dz # one sided difference
    f33[0, :, :] = (Zf[1, :, :] - Zf[0, :, :]) / dz # one sided difference

    f13[-1, :, :] = (Xf[-1, :, :] - Xf[-2, :, :]) / dz # one sided difference
    f23[-1, :, :] = (Yf[-1, :, :] - Yf[-2, :, :]) / dz # one sided difference
    f33[-1, :, :] = (Zf[-1, :, :] - Zf[-2, :, :]) / dz # one sided difference

    # compute the Cauchy_Green tensor F^T . F
    C11 = f11*f11 + f21*f21 + f31*f31
    C12 = f11*f12 + f21*f22 + f31*f32
    C13 = f11*f13 + f21*f23 + f31*f33
    C21 = C12
    C22 = f12*f12 + f22*f22 + f32*f32
    C23 = f12*f13 + f22*f23 + f32*f33
    C31 = C13
    C32 = C23
    C33 = f13*f13 + f23*f23 + f33*f33


    # eigenvalues
    T1 = C11 + C22 + C33 # tr(C)
    T2 = C11*C11 + C22*C22 + C33*C33 + 2*(C12*C12 + C13*C13 + C23*C23) # tr(C^2)
    T3 = C11*(C22*C33 - C23*C23) - C12*(C12*C33 - C13*C23) + C13*(C12*C23 - C13*C22) # det(C)
    Q = (T1*T1 - 3*T2) / 18
    R = (5*T1*T1*T1 - 9*T1*T2 - 54*T3) / 108

    Theta = np.acos( np.clip(R/np.sqrt(-Q*Q*Q), -1, 1) )

    lambda0 = T1/3 + 2*np.sqrt(-Q) * np.cos((Theta + 2*np.pi*0)/3)
    lambda1 = T1/3 + 2*np.sqrt(-Q) * np.cos((Theta + 2*np.pi*1)/3)
    lambda2 = T1/3 + 2*np.sqrt(-Q) * np.cos((Theta + 2*np.pi*2)/3)

    max_lambda = np.maximum.reduce([lambda0, lambda1, lambda2])

    # compute the Lyapunov exponent
    return np.log(max_lambda) / (2*abs(T))


# def main(*, filename: str='/nesi/nobackup/nesi99999/chris.zweck/CSP-224-FTLE/blf_day_loc1_4m_xy_N04.003.nc', 
#     save_dir: str='/nesi/nobackup/nesi99999/chris.zweck/CSP-224-FTLE/test', 
#     t_start: int=1500, t_end: int=1521, T: int=-10,     
#     lcs_dt: int=1, lcs_dx: int=8, lcs_dy: int=8, lcs_dz: int=8, dt: int=1):
def main(*, filename: str='small_blf_day_loc1_4m_xy_N04.003.nc', 
    save_dir: str='./test', 
    t_start: int=10, t_end: int=21, 
    T: int=-10,     
    lcs_dx: int=8, lcs_dy: int=8, lcs_dz: int=8, dt: int=1):
    """
    Compute the FTLE
    @param filename input PALM NetCDF file name
    @param save_dir directory to save the output file
    @param t_start starting time index
    @param t_end one past last time index
    @param T trajectory integration time
    @param lcs_dx? why do we need to specify this? the resolution is 4m in the netcdf file
    @param lcs_dy?
    @param lcs_dz?
    @param dt
    """

    # read the data
    ds = xr.open_dataset(filename, engine='netcdf4', decode_timedelta=False).isel(time=slice(t_start, t_end))

    dx = (ds.x[1] - ds.x[0]).item()
    dy = (ds.y[1] - ds.y[0]).item()
    dz = (ds.zu_xy[1] - ds.zu_xy[0]).item()

    print(f'time = {ds.time}')

    for time_index in range(len(ds.time)):

        ftle = compute_ftle(ds, time_index)


    # new_u = 0.5 * (ds.u_xy[:,:,:, :-1].values + ds.u_xy[:,:,:, 1:].values) #(t, z, y, x)
    # new_v = 0.5 * (ds.v_xy[:,:,:-1, :].values + ds.v_xy[:,:,1:, :].values)

    # new_u = new_u[:,1:,:-1,:] 
    # new_v = new_v[:,1:,:,:-1]

    # #same for w 
    # new_w = 0.5 * (ds.w_xy[:,0:len(ds.zu_xy)-1,:,:].values + ds.w_xy[:,1:len(ds.zu_xy),:,:].values)
    # new_w = new_w[:,:,:-1,:-1] #check axes
    # new_z = ds.zu_xy[1:].values

    # new_x = ds.x[:-1].values
    # new_y = ds.y[:-1].values
    
    # #from (t, z, y, x) to (t, x, y, z)
    # u = new_u.transpose(0, 3, 2, 1)
    # v = new_v.transpose(0, 3, 2, 1)
    # w = new_w.transpose(0, 3, 2, 1)

    # # remove nans (buildings)
    # u = np.nan_to_num(u, nan=0.0)
    # v = np.nan_to_num(v, nan=0.0)
    # w = np.nan_to_num(w, nan=0.0)

    # grid_vel, C_eval_u, C_eval_v, C_eval_w = get_interp_arrays_3D(t, new_x, new_y, new_z, u, v, w)
    # funcptr = get_flow_3D(grid_vel, C_eval_u, C_eval_v, C_eval_w)

    # # create the directory if it does not exist
    # Path(save_dir).mkdir(parents=True, exist_ok=True)

    # for t0 in t: # WHY DO WE START AT 10?

    #     print(f"Processing t0 = {t0}")

    #     rtol = 1e-3
    #     atol = 1e-5

    #     flowmap = flowmap_grid_3D(funcptr, t0, T, new_x, new_y, new_z, params)
    #     ftle = ftle_grid_3D(flowmap, T, lcs_dx, lcs_dy, lcs_dz)

    #     #uncomment to check plot
    #     # plt.imshow(ftle[:,:,4].transpose(), origin='lower')
    #     # plt.colorbar()
    #     # plt.show()

    #     print(f'check sum: {ftle.sum()}')

    #     writeFTLEVTI(ftle, f'{save_dir}/ftle3D_{t_start+t0}.vti')

if __name__ == '__main__':
    defopt.run(main)

