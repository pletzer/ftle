#!/usr/bin/env python3
"""
Parallel FTLE computation (Dask delayed).
Each worker opens the dataset file, selects the single time slice it needs,
computes FTLE and writes the VTI output. This avoids passing large xarray
datasets between processes.

Usage example:
    python ftle_fast_parallel.py --filename data.nc --tmin 0 --tmax 4 --T -10
"""
from __future__ import annotations
import sys
from typing import Optional, Tuple, List

import numpy as np
import xarray as xr
import defopt
import pyvista as pv
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster

from numba import njit

def _prepare_grid_and_faces(
    ds: xr.Dataset,
    time_index: int,
    imin: int,
    imax: Optional[int],
    jmin: int,
    jmax: Optional[int],
) -> dict:
    # interpret None for imax/jmax as "to the end"
    xu_slice = slice(imin, imax)
    yv_slice = slice(jmin, jmax)

    x = np.asarray(ds.xu[xu_slice]).astype(np.float64)
    y = np.asarray(ds.yv[yv_slice]).astype(np.float64)
    z = np.asarray(ds.zw_xy).astype(np.float64)

    if x.size < 2 or y.size < 2 or z.size < 2:
        raise ValueError("Need at least 2 grid points in each direction.")

    xmin, ymin, zmin = x[0], y[0], z[0]
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    nx, ny, nz = len(x), len(y), len(z)

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")  # (nz, ny, nx)

    n = nx * ny * nz
    xflat = xx.ravel()
    yflat = yy.ravel()
    zflat = zz.ravel()

    # NOTE: ds passed to this function must contain a time dimension of length >= 1
    # and the second dimension is elevation (z). Replace Nans with zeros.
    uface = np.asarray(ds.u_xy[time_index, :, jmin:jmax, imin:imax].fillna(0.0)).astype(np.float64)
    vface = np.asarray(ds.v_xy[time_index, :, jmin:jmax, imin:imax].fillna(0.0)).astype(np.float64)
    wface = np.asarray(ds.w_xy[time_index, :, jmin:jmax, imin:imax].fillna(0.0)).astype(np.float64)

    ifloat = (xflat - xmin) / dx
    jfloat = (yflat - ymin) / dy
    kfloat = (zflat - zmin) / dz

    i0 = np.floor(ifloat).astype(np.int64)
    j0 = np.floor(jfloat).astype(np.int64)
    k0 = np.floor(kfloat).astype(np.int64)

    i0 = np.clip(i0, 0, nx - 2)
    j0 = np.clip(j0, 0, ny - 2)
    k0 = np.clip(k0, 0, nz - 2)

    xsi_grid = (ifloat - i0).astype(np.float64)
    eta_grid = (jfloat - j0).astype(np.float64)
    zet_grid = (kfloat - k0).astype(np.float64)

    return dict(
        nx=nx, ny=ny, nz=nz, n=n,
        xmin=xmin, ymin=ymin, zmin=zmin,
        dx=dx, dy=dy, dz=dz,
        x=x, y=y, z=z,
        xx=xx, yy=yy, zz=zz,
        xflat=xflat, yflat=yflat, zflat=zflat,
        i0=i0, j0=j0, k0=k0,
        xsi_grid=xsi_grid, eta_grid=eta_grid, zet_grid=zet_grid,
        uface=uface, vface=vface, wface=wface
    )

def vel_fun_vectorized(
    pos: np.ndarray,
    n: int,
    xmin: float, ymin: float, zmin: float,
    dx: float, dy: float, dz: float,
    nx: int, ny: int, nz: int,
    uface: np.ndarray, vface: np.ndarray, wface: np.ndarray,
) -> np.ndarray:
    xi = pos[0:n]
    yi = pos[n:2*n]
    zi = pos[2*n:3*n]

    ifloat = (xi - xmin) / dx
    jfloat = (yi - ymin) / dy
    kfloat = (zi - zmin) / dz

    ifloat = np.clip(ifloat, 0.0, nx - 1.0)
    jfloat = np.clip(jfloat, 0.0, ny - 1.0)
    kfloat = np.clip(kfloat, 0.0, nz - 1.0)

    i0 = np.clip(np.floor(ifloat).astype(np.int64), 0, nx - 2)
    j0 = np.clip(np.floor(jfloat).astype(np.int64), 0, ny - 2)
    k0 = np.clip(np.floor(kfloat).astype(np.int64), 0, nz - 2)

    xsi = ifloat - i0
    eta = jfloat - j0
    zet = kfloat - k0

    isx = 1.0 - xsi
    ate = 1.0 - eta
    tez = 1.0 - zet

    ui = uface[k0, j0, i0] * isx + uface[k0, j0, i0 + 1] * xsi
    vi = vface[k0, j0, i0] * ate + vface[k0, j0 + 1, i0] * eta
    wi = wface[k0, j0, i0] * tez + wface[k0 + 1, j0, i0] * zet

    return np.concatenate([ui, vi, wi])

# Numba helpers (kept top-level)
@njit(cache=True)
def _vel_fun_numba(
    pos,
    n,
    xmin, ymin, zmin,
    dx, dy, dz,
    nx, ny, nz,
    uface, vface, wface
):
    ui = np.empty(n, dtype=np.float64)
    vi = np.empty(n, dtype=np.float64)
    wi = np.empty(n, dtype=np.float64)

    for idx in range(n):
        x = pos[idx]
        y = pos[n + idx]
        z = pos[2*n + idx]

        ifloat = (x - xmin) / dx
        jfloat = (y - ymin) / dy
        kfloat = (z - zmin) / dz

        ifloat = max(0.0, min(ifloat, nx - 1.0))
        jfloat = max(0.0, min(jfloat, ny - 1.0))
        kfloat = max(0.0, min(kfloat, nz - 1.0))

        i0 = int(np.floor(ifloat))
        j0 = int(np.floor(jfloat))
        k0 = int(np.floor(kfloat))

        if i0 >= nx - 1:
            i0 = nx - 2
        if j0 >= ny - 1:
            j0 = ny - 2
        if k0 >= nz - 1:
            k0 = nz - 2

        xsi = ifloat - i0
        eta = jfloat - j0
        zet = kfloat - k0

        isx = 1.0 - xsi
        ate = 1.0 - eta
        tez = 1.0 - zet

        ui[idx] = uface[k0, j0, i0] * isx + uface[k0, j0, i0 + 1] * xsi
        vi[idx] = vface[k0, j0, i0] * ate + vface[k0, j0 + 1, i0] * eta
        wi[idx] = wface[k0, j0, i0] * tez + wface[k0 + 1, j0, i0] * zet

    out = np.empty(3 * n, dtype=np.float64)
    for i in range(n):
        out[i] = ui[i]
        out[n + i] = vi[i]
        out[2*n + i] = wi[i]
    return out

@njit(cache=True)
def _rk4_numba(y0, t0, t1, nsteps,
               n,
               xmin, ymin, zmin,
               dx, dy, dz,
               nx, ny, nz,
               uface, vface, wface):
    y = y0.copy()
    dt = (t1 - t0) / nsteps
    k1 = np.empty(3*n, dtype=np.float64)
    k2 = np.empty(3*n, dtype=np.float64)
    k3 = np.empty(3*n, dtype=np.float64)
    k4 = np.empty(3*n, dtype=np.float64)
    tmp = np.empty(3*n, dtype=np.float64)

    for step in range(nsteps):
        k1[:] = _vel_fun_numba(y, n, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz, uface, vface, wface)
        for i in range(3*n):
            tmp[i] = y[i] + 0.5*dt*k1[i]
        k2[:] = _vel_fun_numba(tmp, n, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz, uface, vface, wface)
        for i in range(3*n):
            tmp[i] = y[i] + 0.5*dt*k2[i]
        k3[:] = _vel_fun_numba(tmp, n, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz, uface, vface, wface)
        for i in range(3*n):
            tmp[i] = y[i] + dt*k3[i]
        k4[:] = _vel_fun_numba(tmp, n, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz, uface, vface, wface)
        for i in range(3*n):
            y[i] = y[i] + (dt/6.0)*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])
    return y

def estimate_nsteps(uface: np.ndarray, vface: np.ndarray, wface: np.ndarray,
                    dx: float, dy: float, dz: float, T: float, min_steps: int = 20) -> int:
    Umax = float(np.sqrt(uface * uface + vface * vface + wface * wface).max())
    if Umax <= 0:
        return min_steps
    hmin = min(dx, dy, dz)
    crossings = Umax * abs(T) / hmin
    nsteps = max(int(4.0 * crossings) + 1, min_steps)
    return nsteps


import numpy as np

def gradient_corner_to_center(Xf, dx, dy, dz):
    """
    Cell-cantered gradients for a field defined at cell corners.
    Xf has shape (nz+1, ny+1, nx+1) = (k, j, i).

    Returns:
        (dXdx, dXdy, dXdz) each shaped (nz, ny, nx)
    """

    # Corner cube at (k, j, i)
    c000 = Xf[:-1, :-1, :-1]   # (k,   j,   i)
    c100 = Xf[:-1, :-1,  1:]   # (k,   j,   i+1)
    c010 = Xf[:-1,  1:, :-1]   # (k,   j+1, i)
    c110 = Xf[:-1,  1:,  1:]   # (k,   j+1, i+1)

    c001 = Xf[ 1:, :-1, :-1]   # (k+1, j,   i)
    c101 = Xf[ 1:, :-1,  1:]   # (k+1, j,   i+1)
    c011 = Xf[ 1:,  1:, :-1]   # (k+1, j+1, i)
    c111 = Xf[ 1:,  1:,  1:]   # (k+1, j+1, i+1)

    # ----- dX/dx — difference across i -----
    dXdx = 0.25 * (
          (c100 + c110 + c101 + c111)   # +i side
        - (c000 + c010 + c001 + c011)   # -i side
    ) / dx

    # ----- dX/dy — difference across j -----
    dXdy = 0.25 * (
          (c010 + c110 + c011 + c111)   # +j side
        - (c000 + c100 + c001 + c101)   # -j side
    ) / dy

    # ----- dX/dz — difference across k -----
    dXdz = 0.25 * (
          (c001 + c101 + c011 + c111)   # +k side
        - (c000 + c100 + c010 + c110)   # -k side
    ) / dz

    return dXdx, dXdy, dXdz


def compute_ftle(
    ds: xr.Dataset,
    time_index: int,
    T: float,
    imin: int,
    imax: Optional[int],
    jmin: int,
    jmax: Optional[int],
    use_numba: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    if T == 0:
        raise ValueError("Integration time T must be non-zero.")

    ds = ds.load()
    G = _prepare_grid_and_faces(ds, time_index, imin, imax, jmin, jmax)
    nx = G['nx']; ny = G['ny']; nz = G['nz']; n = G['n']
    xmin = G['xmin']; ymin = G['ymin']; zmin = G['zmin']
    dx = G['dx']; dy = G['dy']; dz = G['dz']
    uface = G['uface']; vface = G['vface']; wface = G['wface']
    xflat = G['xflat']; yflat = G['yflat']; zflat = G['zflat']

    if verbose:
        print(f"nx,ny,nz = {nx},{ny},{nz}, npoints = {n}")
        print(f"dx,dy,dz = {dx},{dy},{dz}")

    y0 = np.concatenate([xflat, yflat, zflat]).astype(np.float64)
    nsteps = estimate_nsteps(uface, vface, wface, dx, dy, dz, T)
    if verbose:
        print(f"Using nsteps = {nsteps} for RK4 integration")

    # Compute the trajectories. NOTE: we're freezing the velocity
    # in this version
    t0 = 0.0
    t1 = float(T)

    if verbose:
        print("Using Numba-accelerated RK4 integrator")
    final_pos = _rk4_numba(y0, t0, t1, nsteps,
                            n,
                            xmin, ymin, zmin,
                            dx, dy, dz,
                            nx, ny, nz,
                            uface, vface, wface)

    Xf = final_pos[0:n].reshape((nz, ny, nx))
    Yf = final_pos[n:2*n].reshape((nz, ny, nx))
    Zf = final_pos[2*n:3*n].reshape((nz, ny, nx))

    # Compute the deformation gradient F at cell centres
    f11, f12, f13 = gradient_corner_to_center(Xf, dx, dy, dz)
    f21, f22, f23 = gradient_corner_to_center(Yf, dx, dy, dz)
    f31, f32, f33 = gradient_corner_to_center(Zf, dx, dy, dz)

    # Compute the Cauchy-Green tensor F^T . F
    C11 = f11*f11 + f21*f21 + f31*f31
    C12 = f11*f12 + f21*f22 + f31*f32
    C13 = f11*f13 + f21*f23 + f31*f33
    C22 = f12*f12 + f22*f22 + f32*f32
    C23 = f12*f13 + f22*f23 + f32*f33
    C33 = f13*f13 + f23*f23 + f33*f33

    # Compute the eigenvalues of C
    C = np.empty((nz-1, ny-1, nx-1, 3, 3), dtype=C11.dtype)
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
    eigvals = np.linalg.eigvalsh(C_flat)
    max_lambda = eigvals[:, -1].reshape((nz-1, ny-1, nx-1))

    eps = 1e-16
    max_lambda = np.clip(max_lambda, eps, None)

    ftle = np.log(max_lambda) / (2.0 * abs(float(T)))

    # flte shoiuld be cell centred
    return ftle

# -------------------------
# Worker: runs on each process/thread
# -------------------------
def _worker_compute_and_save(
    ds_path: str,
    time_index: int,
    T: float,
    imin: int,
    imax: Optional[int],
    jmin: int,
    jmax: Optional[int],
    use_numba: bool,
    verbose: bool,
    outdir: Optional[str]
) -> str:
    """
    Opens dataset, selects a single-time slice ds_t with a time dim of length 1,
    computes FTLE (using time_index=0 inside compute_ftle) and saves VTI file.
    Returns the output filename.
    """
    # local imports to keep worker environment self-contained
    import os
    import xarray as xr
    import numpy as np
    import pyvista as pv

    # engine="netcdf4" fails when reading the file in parallel
    ds = xr.open_dataset(ds_path, engine="h5netcdf", decode_timedelta=False)
    # keep a time dimension of length 1 so compute_ftle's indexing works:
    ds_t = ds.isel(time=slice(time_index, time_index + 1))

    # load the required slice into memory (small)
    ds_t = ds_t.load()

    # compute ftle; now the time index inside this sliced dataset is 0
    ftle = compute_ftle(ds_t, 0, T, imin, imax, jmin, jmax, use_numba=use_numba, verbose=verbose)

    nz, ny, nx = ftle.shape
    # ftle is cell centred
    nz += 1
    ny += 1
    nx += 1

    # assemble output grid and save VTI
    dx = float((ds.xu[1] - ds.xu[0]).item())
    dy = float((ds.yv[1] - ds.yv[0]).item())
    dz = float((ds.zw_xy[1] - ds.zw_xy[0]).item())

    ftle_for_output = np.transpose(ftle, (2, 1, 0)).astype(np.float32)
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (dx, dy, dz)

    origin_x = float(ds.xu[imin].item())
    origin_y = float(ds.yv[jmin].item())
    origin_z = float(ds.zw_xy[0].item())
    grid.origin = (origin_x, origin_y, origin_z)

    grid.cell_data['ftle'] = ftle_for_output.ravel(order='F')
    print(f'checksum: {np.abs(ftle_for_output.ravel(order="F")).sum()}')
    if outdir is None:
        outdir = "."
    os.makedirs(outdir, exist_ok=True)
    outname = f"{outdir}/ftle_{time_index:05}.vti"
    grid.save(outname)
    if verbose:
        print(f"[worker] wrote {outname}")
    return outname

# -------------------------
# Main driver: submit tasks to Dask
# -------------------------
def main(
    *,
    filename: str = "small_blf_day_loc1_4m_xy_N04.003.nc",
    tmin: int = 0,
    tmax: int = 1,
    T: float = -10.0,
    imin: int = 0,
    imax: int = -1,
    jmin: int = 0,
    jmax: int = -1,
    use_numba: bool = True,
    verbose: bool = False,
    outdir: Optional[str] = None,
    n_workers: int = 1,
    threads_per_worker: int = 2,
):
    """
    Submit FTLE tasks for time indices in [tmin, tmax) to Dask.
    Each task loads only its time slice and writes a VTI file.
    """
    if imax < 0:
        imax_val = None
    else:
        imax_val = imax
    if jmax < 0:
        jmax_val = None
    else:
        jmax_val = jmax

    # Build delayed tasks (do not open the dataset here)
    tasks = []
    for ti in range(tmin, tmax):
        task = delayed(_worker_compute_and_save)(
            filename, ti, float(T),
            imin, imax_val, jmin, jmax_val,
            use_numba, verbose, outdir
        )
        tasks.append(task)

    # Trigger computation in parallel
    if verbose:
        print(f"Submitting {len(tasks)} tasks to dask client")

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)

    results = dask.compute(*tasks, scheduler=client)
    if verbose:
        print("Completed tasks. Outputs:")
        for r in results:
            print("  ", r)

if __name__ == "__main__":
    defopt.run(main)
