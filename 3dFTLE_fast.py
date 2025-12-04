#!/usr/bin/env python3
"""
Faster FTLE computation adapted from user's original script.

Features:
- Precomputes cell indices and fractional offsets for interpolation
- Vectorized RK4 integrator (numpy) + optional Numba-jitted integrator
- Better RK4 step estimate (CFL-like)
- Uses np.gradient for deformation gradient
- Correct output ordering for pyvista ImageData
- Verbosity flag to control prints

Usage example (CLI via defopt):
    python ftle_fast.py --filename small_blf_day_loc1_4m_xy_N04.003.nc --t_start 10 --t_end 11 --T -10
"""

from __future__ import annotations
import sys
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import defopt
import pyvista as pv

# Optional numba
try:
    import numba as _numba
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# -------------------------
# Utility / interpolation
# -------------------------
def _prepare_grid_and_faces(
    ds: xr.Dataset,
    time_index: int,
    imin: int,
    imax: int,
    jmin: int,
    jmax: int,
) -> Tuple:
    """
    Read x/y/z coordinates and face velocities for the patch and
    precompute flattened mesh + integer indices and fractional offsets.

    Returns:
        dict with keys:
            nx, ny, nz, n, xmin, ymin, zmin, dx, dy, dz,
            x, y, z, xx, yy, zz,
            i0, j0, k0,
            uface, vface, wface
    """
    # Select coordinates (nodal positions)
    x = np.asarray(ds.xu[imin:imax]).astype(np.float64)
    y = np.asarray(ds.yv[jmin:jmax]).astype(np.float64)
    z = np.asarray(ds.zw_xy).astype(np.float64)

    if x.size < 2 or y.size < 2 or z.size < 2:
        raise ValueError("Need at least 2 grid points in each direction.")

    xmin, ymin, zmin = x[0], y[0], z[0]
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    nx, ny, nz = len(x), len(y), len(z)
    # mesh with indexing 'ij' so shapes are (nz, ny, nx)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")  # (nz, ny, nx)

    # flatten
    n = nx * ny * nz
    xflat = xx.ravel()
    yflat = yy.ravel()
    zflat = zz.ravel()

    # read faces at time_index and clip NaNs -> 0
    uface = np.asarray(ds.u_xy[time_index, :, jmin:jmax, imin:imax].fillna(0.0)).astype(np.float64)
    vface = np.asarray(ds.v_xy[time_index, :, jmin:jmax, imin:imax].fillna(0.0)).astype(np.float64)
    wface = np.asarray(ds.w_xy[time_index, :, jmin:jmax, imin:imax].fillna(0.0)).astype(np.float64)

    # Precompute base integer indices (lower corner) and fractional offsets for the grid points.
    # For particles that will move off-grid we will clamp indices during interpolation.
    ifloat = (xflat - xmin) / dx
    jfloat = (yflat - ymin) / dy
    kfloat = (zflat - zmin) / dz

    # floor gives lower cell index; clamp to valid [0, size-2]
    i0 = np.floor(ifloat).astype(np.int64)
    j0 = np.floor(jfloat).astype(np.int64)
    k0 = np.floor(kfloat).astype(np.int64)

    i0 = np.clip(i0, 0, nx - 2)
    j0 = np.clip(j0, 0, ny - 2)
    k0 = np.clip(k0, 0, nz - 2)

    # fractional part inside cell in [0,1)
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

# -------------------------
# Vectorized velocity evaluator (numpy)
# -------------------------
def vel_fun_vectorized(
    pos: np.ndarray,
    n: int,
    xmin: float, ymin: float, zmin: float,
    dx: float, dy: float, dz: float,
    nx: int, ny: int, nz: int,
    uface: np.ndarray, vface: np.ndarray, wface: np.ndarray,
) -> np.ndarray:
    """
    pos: length 3*n vector [x..., y..., z...]
    returns concatenated [u..., v..., w...] length 3*n
    Uses trilinear-like interpolation tailored for face-centered fields:
      - u varies in x between i0 and i0+1 at face centers (assumed)
      - v varies in y
      - w varies in z
    This is a vectorized numpy implementation.
    """
    xi = pos[0:n]
    yi = pos[n:2*n]
    zi = pos[2*n:3*n]

    # fractional coordinates in cell
    ifloat = (xi - xmin) / dx
    jfloat = (yi - ymin) / dy
    kfloat = (zi - zmin) / dz

    # clamp (so particle leaving domain will be evaluated at boundary cell)
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

    # ui: linear in x between i0 and i0+1 at the same (k0,j0)
    ui = uface[k0, j0, i0] * isx + uface[k0, j0, i0 + 1] * xsi
    # vi: linear in y between j0 and j0+1 at same (k0,i0)
    vi = vface[k0, j0, i0] * ate + vface[k0, j0 + 1, i0] * eta
    # wi: linear in z between k0 and k0+1 at same (j0,i0)
    wi = wface[k0, j0, i0] * tez + wface[k0 + 1, j0, i0] * zet

    return np.concatenate([ui, vi, wi])


# -------------------------
# Numba implementation (optional)
# -------------------------
# We provide a numba-jitted velocity + RK4 integrator if numba is available.
if NUMBA_AVAILABLE:
    from numba import njit, prange

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

        # pos layout [x..., y..., z...]
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

    @njit(cache=True, parallel=False)
    def _rk4_numba(y0, t0, t1, nsteps,
                   n,
                   xmin, ymin, zmin,
                   dx, dy, dz,
                   nx, ny, nz,
                   uface, vface, wface):
        """
        Numba RK4 (in-place), returns final y array (length 3*n)
        """
        y = y0.copy()
        dt = (t1 - t0) / nsteps
        # allocate temporaries
        k1 = np.empty(3*n, dtype=np.float64)
        k2 = np.empty(3*n, dtype=np.float64)
        k3 = np.empty(3*n, dtype=np.float64)
        k4 = np.empty(3*n, dtype=np.float64)
        tmp = np.empty(3*n, dtype=np.float64)

        t = t0
        for step in range(nsteps):
            k1[:] = _vel_fun_numba(y, n, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz, uface, vface, wface)

            # tmp = y + 0.5*dt*k1
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

            t += dt
        return y

# -------------------------
# RK4 step estimate (CFL-like)
# -------------------------
def estimate_nsteps(uface: np.ndarray, vface: np.ndarray, wface: np.ndarray,
                    dx: float, dy: float, dz: float, T: float, min_steps: int = 20) -> int:
    """
    Estimate number of RK4 steps using a CFL-like heuristic:
      nsteps ~ 4 * (Umax * |T| / hmin)
    with lower bound min_steps.
    """
    Umax = float(np.sqrt(uface * uface + vface * vface + wface * wface).max())
    if Umax <= 0:
        return min_steps
    hmin = min(dx, dy, dz)
    crossings = Umax * abs(T) / hmin
    nsteps = max(int(4.0 * crossings) + 1, min_steps)
    return nsteps

# -------------------------
# Compute FTLE
# -------------------------
def compute_ftle(
    ds: xr.Dataset,
    time_index: int,
    T: float,
    imin: int,
    imax: int,
    jmin: int,
    jmax: int,
    use_numba: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute FTLE for the requested patch.

    Returns ftle shaped (nz, ny, nx) just like internal arrays; caller may transpose for output.
    """
    if T == 0:
        raise ValueError("Integration time T must be non-zero.")

    # ensure data in memory
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

    # initial concatenated positions
    y0 = np.concatenate([xflat, yflat, zflat]).astype(np.float64)

    # choose integrator steps
    nsteps = estimate_nsteps(uface, vface, wface, dx, dy, dz, T)
    if verbose:
        print(f"Using nsteps = {nsteps} for RK4 integration")

    t0 = 0.0
    t1 = float(T)

    # pick integrator: prefer numba if available and requested
    if NUMBA_AVAILABLE and use_numba:
        if verbose:
            print("Using Numba-accelerated RK4 integrator")
        final_pos = _rk4_numba(y0, t0, t1, nsteps,
                               n,
                               xmin, ymin, zmin,
                               dx, dy, dz,
                               nx, ny, nz,
                               uface, vface, wface)
    else:
        if verbose and use_numba:
            print("Numba not available or disabled; using numpy RK4")
        # numpy RK4: keep temporaries to avoid extra allocations in loop
        y = y0.copy()
        dt = (t1 - t0) / nsteps
        k1 = np.empty_like(y)
        k2 = np.empty_like(y)
        k3 = np.empty_like(y)
        k4 = np.empty_like(y)
        tmp = np.empty_like(y)

        for step in range(nsteps):
            k1[:] = vel_fun_vectorized(y, n, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz, uface, vface, wface)
            tmp[:] = y + 0.5 * dt * k1
            k2[:] = vel_fun_vectorized(tmp, n, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz, uface, vface, wface)
            tmp[:] = y + 0.5 * dt * k2
            k3[:] = vel_fun_vectorized(tmp, n, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz, uface, vface, wface)
            tmp[:] = y + dt * k3
            k4[:] = vel_fun_vectorized(tmp, n, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz, uface, vface, wface)

            y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        final_pos = y

    # reshape to (nz, ny, nx)
    Xf = final_pos[0:n].reshape((nz, ny, nx))
    Yf = final_pos[n:2*n].reshape((nz, ny, nx))
    Zf = final_pos[2*n:3*n].reshape((nz, ny, nx))

    # compute gradients using np.gradient (gives arrays same shape)
    # note np.gradient expects axes order consistent with array axes
    dXdz, dXdy, dXdx = np.gradient(Xf, dz, dy, dx, edge_order=2)
    dYdz, dYdy, dYdx = np.gradient(Yf, dz, dy, dx, edge_order=2)
    dZdz, dZdy, dZdx = np.gradient(Zf, dz, dy, dx, edge_order=2)

    # f_ij = d{X,Y,Z}/d{x,y,z}
    f11 = dXdx; f12 = dXdy; f13 = dXdz
    f21 = dYdx; f22 = dYdy; f23 = dYdz
    f31 = dZdx; f32 = dZdy; f33 = dZdz

    # Cauchy-Green tensor components
    C11 = f11*f11 + f21*f21 + f31*f31
    C12 = f11*f12 + f21*f22 + f31*f32
    C13 = f11*f13 + f21*f23 + f31*f33
    C22 = f12*f12 + f22*f22 + f32*f32
    C23 = f12*f13 + f22*f23 + f32*f33
    C33 = f13*f13 + f23*f23 + f33*f33

    # build 3x3 matrices and compute max eigenvalue per point
    C = np.empty((nz, ny, nx, 3, 3), dtype=C11.dtype)
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
    max_lambda = eigvals[:, -1].reshape((nz, ny, nx))

    # numerical safety
    eps = 1e-16
    max_lambda = np.clip(max_lambda, eps, None)

    ftle = np.log(max_lambda) / (2.0 * abs(float(T)))

    return ftle


# -------------------------
# Main CLI
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
):
    """
    Compute FTLE on a patch from a PALM NetCDF file.
    CLI example: python ftle_fast.py --filename data.nc --tmin 0 --tmax 1 --T -10
    """
    ds = xr.open_dataset(filename, engine="netcdf4", decode_timedelta=False)

    # interpret negative imax/jmax as python slice semantics (like your original)
    if imax < 0:
        imax = None
    if jmax < 0:
        jmax = None

    # Preload coords for spacing (you used ds.xu etc)
    dx = float((ds.xu[1] - ds.xu[0]).item())
    dy = float((ds.yv[1] - ds.yv[0]).item())
    dz = float((ds.zw_xy[1] - ds.zw_xy[0]).item())

    if verbose:
        print(f"dataset time coords: {ds.time}")
        print(f"grid spacing dx,dy,dz = {dx},{dy},{dz}")

    for time_index in range(tmin, tmax):
        if verbose:
            print(f"\nComputing FTLE for time_index = {time_index}")
        ftle = compute_ftle(
            ds, time_index, T=float(T),
            imin=imin, imax=imax, jmin=jmin, jmax=jmax,
            use_numba=use_numba,
            verbose=verbose
        )

        nz, ny, nx = ftle.shape
        if verbose:
            print(f"ftle shape (nz,ny,nx) = {ftle.shape}")
            print(f"abs sum = {np.fabs(ftle).sum():.6e}")
            # check that ftle is non-zero at the lowest elevation
            print(f'ftle[0,...] = {ftle[0,...]}')

        # PyVista expects point data in x-fastest order depending on dims; convert to (nx,ny,nz)
        # Our ftle is (nz,ny,nx) -> transpose to (nx,ny,nz)
        ftle_for_output = np.transpose(ftle, (2, 1, 0)).astype(np.float32)

        grid = pv.ImageData()
        grid.dimensions = (nx, ny, nz)
        grid.spacing = (dx, dy, dz)
        grid.origin = (ds.xu[imin], ds.yv[jmin], ds.zw_xy[0])
        # pyvista expects point data length = nx*ny*nz
        grid.point_data['ftle'] = ftle_for_output.ravel(order='F')  # Fortran order matches (x,y,z) ordering
        outname = f"ftle_{time_index:05}.vti"
        grid.save(outname)
        if verbose:
            print(f"Wrote {outname}")

if __name__ == "__main__":
    defopt.run(main)
