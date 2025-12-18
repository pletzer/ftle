"""
Custom ParaView Python Source plugin to read and compute the Finite Time Lyapunov Exponent 
from PALM data stored in a NetCDF file. This version allows the velocity field to vary in time
as the grid positions are integrated. 

Inputs:
  - palmfile: path to a NetCDF file
  - tintegr: integration time (float)
  - imin, imax: x-index bounds
  - jmin, jmax: y-index bounds
  - tindex: time index

Reads fields:
  - u_xy, v_xy, w_xy

Grid:
  - Assumed 3D, cell-centred output
  - Index order assumed (time, k, j, i) = (time, nz, ny, nx)
"""


# ----------------------------------------------------------------------
# Usage notes:
# ----------------------------------------------------------------------
# 1. Save this file as e.g. PalmFtleSource.py
# 2. In ParaView:
#    Tools -> Manage Plugins -> Load New -> select this file
# 3. Add source: Sources -> FTLEPythonSource
# 4. Set PalmFile, IntegrationTime, imin/imax, jmin/jmax
# 5. Insert your FTLE kernel where indicated
# ----------------------------------------------------------------------


from paraview.util.vtkAlgorithm import *
import numpy as np
import netCDF4
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkMultiBlockDataSet
import vtk

try:
    # paraview 6.x
    from vtkmodules.util import numpy_support
except:
    from vtk.util import numpy_support

# -------------------------
# RK4 step estimate (CFL-like)
# -------------------------
def _estimate_nsteps(uface: np.ndarray, vface: np.ndarray, wface: np.ndarray,
                    dx: float, dy: float, dz: float, T: float, min_steps: int = 20) -> int:
    """
    Estimate number of RK4 steps using a CFL-like heuristic:
    nsteps ~ 4 * (Umax * |T| / hmin)
    with lower bound min_steps.
    """
    Umax = np.max(np.sqrt(uface**2 + vface**2 + wface**2))
    hmin = min(dx, dy, dz)
    crossings = Umax * abs(T) / hmin
    nsteps = max(int(4.0 * crossings) + 1, min_steps)
    return nsteps

# -------------------------------------
# Cell centred gradient from point data
# -------------------------------------
def _gradient_corner_to_center(Xf: np.ndarray, dx: float, dy: float, dz: np.ndarray) -> np.ndarray:
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
    ) / dz[:, None, None]

    return dXdx, dXdy, dXdz


@smproxy.source(
    name="PalmFtleSource",
    label="PALM FTLE Source",
)
class PalmFtleSource(VTKPythonAlgorithmBase):

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self,
            nInputPorts=0,
            nOutputPorts=1,
            outputType='vtkMultiBlockDataSet' # vtkImageData cannot be used because it needs the extent known ahead of time
        )

        # ---- user parameters (with defaults) ----
        self.palmfile = ""
        self.tintegr = -10.0
        self.imin = 0
        self.imax = 1
        self.jmin = 0
        self.jmax = 1
        self.time_index = 0

    # ------------------------------------------------------------------
    # Properties exposed to ParaView GUI
    # ------------------------------------------------------------------

    @smproperty.stringvector(name="PalmFile", number_of_elements=1, default_values=["/Users/apletzer/work/ftle/paraview_plugin/small_blf_day_loc1_4m_xy_N04.003.nc"])
    @smdomain.filelist()
    @smhint.filechooser(extensions="nc", file_description="NetCDF files")
    def SetPalmFile(self, value):
        # ParaView may pass a string or a list
        if isinstance(value, (list, tuple)):
            self.palmfile = value[0] if value else ""
        else:
            self.palmfile = value
        self.Modified()

    # scalar is a one element vector
    @smproperty.doublevector(name="IntegrationTime", number_of_elements=1, default_values=[-10.0])
    def SetIntegrationTime(self, value, *args):
        self.tintegr = float(value)
        self.Modified()

    @smproperty.intvector(name="TIndex", number_of_elements=1, default_values=[10])
    def SetTIndex(self, value, *args):
        self.time_index = int(value)
        self.Modified()

    @smproperty.intvector(name="IMin", number_of_elements=1, default_values=[180])
    def SetIMin(self, value, *args):
        self.imin = int(value)
        self.Modified()

    @smproperty.intvector(name="IMax", number_of_elements=1, default_values=[200])
    def SetIMax(self, value, *args):
        self.imax = int(value)
        self.Modified()

    @smproperty.intvector(name="JMin", number_of_elements=1, default_values=[220])
    def SetJMin(self, value, *args):
        self.jmin = int(value)
        self.Modified()

    @smproperty.intvector(name="JMax", number_of_elements=1, default_values=[300])
    def SetJMax(self, value, *args):
        self.jmax = int(value)
        self.Modified()


    # ------------------------------------------------------------------
    # Core pipeline method
    # ------------------------------------------------------------------

    def RequestData(self, request, inInfo, outInfo):

        if not self.palmfile:
            raise RuntimeError("PalmFile must be specified")

        res = self._compute_ftle()

        # Axes
        x, y, z = res['x'], res['y'], res['z']
        print(f'x = {x} y = {y} z = {z}')
        # Number of nodes
        nx1, ny1, nz1 = x.shape[0], y.shape[0], z.shape[0]

        # Build image
        grid = vtkRectilinearGrid()
        grid.SetDimensions(nx1, ny1, nz1)

        # convert the numpy arrays to VTK arrays
        x_arr = numpy_support.numpy_to_vtk(num_array=x, deep=True, array_type=vtk.VTK_DOUBLE)
        y_arr = numpy_support.numpy_to_vtk(num_array=y, deep=True, array_type=vtk.VTK_DOUBLE)
        z_arr = numpy_support.numpy_to_vtk(num_array=z, deep=True, array_type=vtk.VTK_DOUBLE)
        grid.SetXCoordinates(x_arr)
        grid.SetYCoordinates(y_arr)
        grid.SetZCoordinates(z_arr)

        # ---- FTLE is cell-centered and currently in (z, y, x) = (17, 80, 20) ----
        # Convert to (x, y, z) = (20, 80, 17)
        ftle_xyz = res['ftle'].transpose((2, 1, 0)).astype(np.float32)  # (nx-1, ny-1, nz-1)

        vtk_arr = numpy_support.numpy_to_vtk(
            num_array=ftle_xyz.ravel(order='F'),   # x fastest, then y, then z
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
        vtk_arr.SetName("FTLE")

        cd = grid.GetCellData()
        cd.AddArray(vtk_arr)
        cd.SetScalars(vtk_arr)  # make FTLE the active cell scalar

        # 3. Put it in the multi-block output
        output = vtkMultiBlockDataSet.GetData(outInfo, 0)
        output.SetBlock(0, grid)
 
        return 1
   
    def select_time_window(self, dt: float, nt: int) -> tuple:
        # --------------------------------------------------------------
        # Select the time window to read data from
        # --------------------------------------------------------------
        di = int(np.ceil(abs(self.tintegr) / dt))
        
        if self.tintegr < 0:
            tmin = max(self.time_index - di, 0)
            tmax = self.time_index + 1
        elif self.tintegr > 0:
            tmin = self.time_index
            tmax = min(self.time_index + di + 1, nt)
        else:
            raise ValueError("tintegr cannot be zero!")

        print(f'self.time_index={self.time_index} dt={dt} nt={nt} tmin={tmin} tmax={tmax}')
        return tmin, tmax


    def get_lower_time_index_and_param_coord(self, time_val: float, t_axis: np.ndarray) -> tuple:
        # --------------------------------------------------------------
        # Get the time index and time interval parametric coordinate from the time value
        # --------------------------------------------------------------

        t_axis_min = np.min(t_axis)
        t_axis_max = np.max(t_axis)
        dt = t_axis[1] - t_axis[0] # assume uniform
        nt = t_axis.shape[0]

        t_index = int(np.floor((time_val - t_axis_min) / dt))
        t_index = np.clip(t_index, 0, nt - 2)

        mu = (time_val - t_axis[t_index])/dt
        mu = np.clip(mu, 0.0, 1.0)

        return (t_index, mu)


    def _compute_ftle(self) -> dict:

        # --------------------------------------------------------------
        # Read NetCDF data
        # --------------------------------------------------------------
        with netCDF4.Dataset(self.palmfile, "r") as nc:

            print(f'self.imin={self.imin} self.imax={self.imax} self.jmin={self.jmin} self.jmax={self.jmax}')

            # axes
            xaxis = nc.variables['xu'][self.imin:self.imax+1]
            yaxis = nc.variables['yv'][self.jmin:self.jmax+1]
            zaxis = nc.variables['zw_xy'][:] # all the elevations
            dt = nc.variables['time'][1] - nc.variables['time'][0] # assume constant time step
            nt_all = nc.variables['time'].size

            tmin, tmax = self.select_time_window(dt, nt_all) # tmin and tmax are indices
            t_axis = nc.variables['time'][tmin:tmax]
            nt = t_axis.shape[0]

            nx_tot = nc.variables['xu'].size
            ny_tot = nc.variables['yv'].size
            if self.imin < 0 or self.imax >= nx_tot:
                raise ValueError("Invalid IRange")
            if self.jmin < 0 or self.jmax >= ny_tot:
                raise ValueError("Invalid JRange")

            xmin = xaxis[0]
            ymin = yaxis[0]
            # assume uniform grid in x, y
            dx = xaxis[1] - xaxis[0]
            dy = yaxis[1] - yaxis[0]
            dz = np.diff(zaxis) # not uniform
            nx1 = len(xaxis)
            ny1 = len(yaxis)
            nz1 = len(zaxis)
            # number of cells
            nx, ny, nz = nx1 - 1, ny1 - 1, nz1 - 1

            # mesh with indexing 'ij' so shapes are (nz, ny, nx)
            zz, yy, xx = np.meshgrid(zaxis, yaxis, xaxis, indexing="ij")
            xflat = xx.ravel()
            yflat = yy.ravel()
            zflat = zz.ravel()

            # read the velocity, expect shape (time, nz, ny, nx). Note we're reading in one more cell in y and
            # x, and all the cells in z. We're also replacing all the nans with zeros
            uface = np.nan_to_num( nc.variables['u_xy'][tmin:tmax, :, self.jmin:self.jmax+1, self.imin:self.imax+1], copy=False, nan=0.0)
            vface = np.nan_to_num( nc.variables['v_xy'][tmin:tmax, :, self.jmin:self.jmax+1, self.imin:self.imax+1], copy=False, nan=0.0)
            wface = np.nan_to_num( nc.variables['w_xy'][tmin:tmax, :, self.jmin:self.jmax+1, self.imin:self.imax+1], copy=False, nan=0.0)
            print(f'nx1={nx1} ny1={ny1} nz1={nz1}')
            print(f'uface.shape={uface.shape}\nvface={vface.shape}\nwface={wface.shape}')

            # total number of grid points
            n = len(xflat)

            def velocity_fun(time_val: float, pos: np.ndarray) -> np.ndarray:
                """
                Velocity interpolated at pos
                pos: length 3*n vector [x..., y..., z...]
                """
                xi = pos[0:n]
                yi = pos[n:2*n]
                zi = pos[2*n:3*n]
                ifloat = (xi - xmin) / dx
                jfloat = (yi - ymin) / dy
                # clamp (so particle leaving domain will be evaluated at boundary cell)
                ifloat = np.clip(ifloat, 0.0, nx1 - 1.0)
                jfloat = np.clip(jfloat, 0.0, ny1 - 1.0)
                # make sure i0, j0 and k0 are on the low side
                i0 = np.clip(np.floor(ifloat).astype(np.int64), 0, nx1 - 2)
                j0 = np.clip(np.floor(jfloat).astype(np.int64), 0, ny1 - 2)
                k0 = np.clip(np.searchsorted(zaxis, zi) - 1, 0, nz1 - 2)

                # parametric coordinates of the cell: 0 <= xsi, eta, zet <= 1
                xsi = ifloat - i0
                eta = jfloat - j0
                zet = (zi - zaxis[k0]) / (zaxis[k0 + 1] - zaxis[k0])

                isx = 1.0 - xsi
                ate = 1.0 - eta
                tez = 1.0 - zet

                # Arakawa C mimetic interpolation of vector field
                # The interpolation is piecewise linear in the direction of the component 
                # and piecewise constant in the other directions. This interpolation
                # conserves fluxes and allows one to have obstacles in the domain (e.g 
                # buildings)

                time_index0, mu = self.get_lower_time_index_and_param_coord(time_val=time_val, t_axis=t_axis)
                # must be well inside
                if time_index0 == nt -1:
                    time_index0 = nt - 2
                    mu = 0.0
                time_index1 = time_index0 + 1

                # u: linear in x between i0 and i0+1 at the same (k0,j0)
                u0 = uface[time_index0, k0, j0, i0] * isx + uface[time_index0, k0, j0, i0 + 1] * xsi
                u1 = uface[time_index1, k0, j0, i0] * isx + uface[time_index1, k0, j0, i0 + 1] * xsi
                # vi: linear in y between j0 and j0+1 at same (k0,i0)
                v0 = vface[time_index0, k0, j0, i0] * ate + vface[time_index0, k0, j0 + 1, i0] * eta
                v1 = vface[time_index1, k0, j0, i0] * ate + vface[time_index1, k0, j0 + 1, i0] * eta
                # wi: linear in z between k0 and k0+1 at same (j0,i0)
                w0 = wface[time_index0, k0, j0, i0] * tez + wface[time_index0, k0 + 1, j0, i0] * zet
                w1 = wface[time_index1, k0, j0, i0] * tez + wface[time_index1, k0 + 1, j0, i0] * zet

                # now time interpolate
                um = 1.0 - mu
                ui = um*u0 + mu*u1
                vi = um*v0 + mu*v1
                wi = um*w0 + mu*w1

                return np.concatenate([ui, vi, wi])          

            # integrate the trajectories. y is the state, a concatenated array of 
            # [x..., y..., z...] positions. xyz0 is the initial condition
            # Note: FTLE is computed from corner-seeded trajectories.
            xyz0 = np.concatenate([xflat, yflat, zflat]).astype(np.float64)
            nsteps = _estimate_nsteps(uface, vface, wface, dx, dy, dz.min(), self.tintegr)

            # Runge-Kutta 4
            xyz = xyz0.copy()
            k1 = np.empty_like(xyz)
            k2 = np.empty_like(xyz)
            k3 = np.empty_like(xyz)
            k4 = np.empty_like(xyz)
            tmp = np.empty_like(xyz)

            local_index = self.time_index - tmin
            time_base = t_axis[local_index]
            delta_time = self.tintegr / nsteps

            for step in range(nsteps):

                time_val = time_base + step * delta_time

                k1[:] = velocity_fun(time_val, xyz)

                time_val += 0.5*delta_time
                tmp[:] = xyz + 0.5 * delta_time * k1
                k2[:] = velocity_fun(time_val, tmp)

                tmp[:] = xyz + 0.5 * delta_time * k2
                k3[:] = velocity_fun(time_val, tmp)

                time_val += 0.5*delta_time
                tmp[:] = xyz + delta_time * k3
                k4[:] = velocity_fun(time_val, tmp)

                xyz += (delta_time / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

                # Make sure the trajectories are not leaving the domain
                # Another possibility would be to have a larger domain for
                # the vector field interpolation. So imin:imax and jmin:jmax 
                # would only be used for seeding the trajectories
                xyz[0:n]   = np.clip(xyz[0:n],   xaxis[0], xaxis[-1])
                xyz[n:2*n] = np.clip(xyz[n:2*n], yaxis[0], yaxis[-1])
                xyz[2*n:]  = np.clip(xyz[2*n:],  zaxis[0], zaxis[-1])

            # reshape
            Xf = xyz[0:n].reshape((nz1, ny1, nx1))
            Yf = xyz[n:2*n].reshape((nz1, ny1, nx1))
            Zf = xyz[2*n:3*n].reshape((nz1, ny1, nx1))

            # Compute the deformation gradient F at cell centres
            f11, f12, f13 = _gradient_corner_to_center(Xf, dx, dy, dz)
            f21, f22, f23 = _gradient_corner_to_center(Yf, dx, dy, dz)
            f31, f32, f33 = _gradient_corner_to_center(Zf, dx, dy, dz)

            # Cauchy-Green tensor components
            C11 = f11*f11 + f21*f21 + f31*f31
            C12 = f11*f12 + f21*f22 + f31*f32
            C13 = f11*f13 + f21*f23 + f31*f33
            C22 = f12*f12 + f22*f22 + f32*f32
            C23 = f12*f13 + f22*f23 + f32*f33
            C33 = f13*f13 + f23*f23 + f33*f33

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

            # Note: the eigenvalues are cell centred (nz, ny, nx)
            max_lambda = np.maximum(eigvals[:, -1], 1.e-16).reshape((nz, ny, nx))

            ftle = np.log(max_lambda) / (2.0 * abs(float(self.tintegr)))
        
            return dict(
                x=xaxis, y=yaxis, z=zaxis, # axes
                ftle=ftle,
        )
