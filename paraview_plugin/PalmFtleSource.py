"""
Custom ParaView Python Source plugin to read and compute the Finite Time Lyapunov Exponent 
from PALM data stored in a NetCDF file

Inputs:
  - palmfile: path to a NetCDF file
  - tintegr: integration time (float)
  - imin, imax: x-index bounds
  - jmin, jmax: y-index bounds
  - tindex: time index

Reads fields:
  - u_xy, v_xy, w_xy

Grid:
  - Assumed uniform, 3D, cell-centred output
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
import vtk

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
    Umax = float(np.sqrt(uface * uface + vface * vface + wface * wface).max())
    if Umax <= 0:
        return min_steps
    hmin = min(dx, dy, dz)
    crossings = Umax * abs(T) / hmin
    nsteps = max(int(4.0 * crossings) + 1, min_steps)
    return nsteps


def _gradient_corner_to_center(Xf, dx, dy, dz):
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


@smproxy.source(
    name="PalmFtleSource",
    label="PALM FTLE Source",
    category="FTLE"
)
@smproperty.xml("""
<SourceProxy>
  <OutputPort name="Output"/>
</SourceProxy>
""")
class PalmFtleSource(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self,
            nInputPorts=0,
            nOutputPorts=1,
            outputType='vtkImageData'
        )

        # ---- user parameters (with defaults) ----
        self.palmfile = ""
        self.tintegr = 1.0
        self.imin = 0
        self.imax = 1
        self.jmin = 0
        self.jmax = 1
        self.tindex = 0

    # ------------------------------------------------------------------
    # Properties exposed to ParaView GUI
    # ------------------------------------------------------------------

    @smproperty.stringvector(name="PalmFile", number_of_elements=1)
    @smdomain.filelist()
    def SetPalmFile(self, value):
        self.palmfile = value[0]
        self.Modified()

    # scalar is a one element vector
    @smproperty.doublevector(name="IntegrationTime", number_of_elements=1, default_values=[1.0])
    def SetIntegrationTime(self, value, *args):
        self.tintegr = float(value)
        self.Modified()

    @smproperty.intvector(name="IRange", number_of_elements=2, default_values=[0, 1])
    def SetIRange(self, value, *args):
        if isinstance(value, (list, tuple)):
            self.imin, self.imax = map(int, value)
        else:
            # single value
            self.imin = int(value)
            self.imax = self.imin + 1
        self.Modified()

    @smproperty.intvector(name="JRange", number_of_elements=2, default_values=[0, 1])
    def SetJRange(self, value, *args):
        if isinstance(value, (list, tuple)):
            self.jmin, self.jmax = map(int, value)
        else:
            # single value
            self.jmin = int(value)
            self.jmax = self.jmin + 1
        self.Modified()

    @smproperty.intvector(name="TIndex", number_of_elements=1, default_values=[0])
    def SetTIndex(self, value, *args):
        self.tindex = int(value)
        self.Modified()

    # ------------------------------------------------------------------
    # Core pipeline method
    # ------------------------------------------------------------------

    def RequestData(self, request, inInfo, outInfo):
        if not self.palmfile:
            raise RuntimeError("PalmFile must be specified")
        
        res = self._compute_ftle()

        # number of nodes in each direction
        nx1, ny1, nz1 = res['nx1'], res['ny1'], res['nz1']
        xmin, ymin, zmin = res['xmin'], res['ymin'], res['zmin']
        dx, dy, dz = res['dx'], res['dy'], res['dz']

        # build the grid
        output = vtk.vtkImageData.GetData(outInfo, 0)
        # number of nodes in x, y and z
        output.SetDimensions(nx1, ny1, nz1)
        output.SetOrigin(xmin, ymin, zmin)
        output.SetSpacing(dx, dy, dz)  # uniform grid assumed

        # (nz,ny,nx) -> transpose to (nx,ny,nz) which is the VTK layout
        ftle = res['ftle'].transpose((2, 1, 0)).astype(np.float32)

        vtk_array = vtk.util.numpy_support.numpy_to_vtk(
            num_array=ftle.ravel(order='F'),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
        vtk_array.SetName("FTLE")

        output.GetCellData().AddArray(vtk_array)
        output.GetCellData().SetScalars(vtk_array)

        return 1
    

    def _compute_ftle(self) -> dict:

        # --------------------------------------------------------------
        # Read NetCDF data
        # --------------------------------------------------------------
        with netCDF4.Dataset(self.palmfile, "r") as nc:

            x = nc.variables['xu'][self.imin:self.imax+1]
            y = nc.variables['yv'][self.jmin:self.jmax+1]
            z = nc.variables['z_xy'][:]
            xmin = x[0]
            ymin = y[0]
            zmin = z[0]
            # assume uniform grid
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            dz = z[1] - z[0]
            nx1 = len(x)
            ny1 = len(y)
            nz1 = len(z)
            # number of cells
            nx, ny, nz = nx1 - 1, ny1 - 1, nz1 - 1

            # mesh with indexing 'ij' so shapes are (nz, ny, nx)
            zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
            xflat = xx.ravel()
            yflat = yy.ravel()
            zflat = zz.ravel()

            # read the velocity, expect shape (time, nz, ny, nx). Note we're reading in one more cell in y and
            # x, and all the cells in z
            uface = nc.variables['u_xy'][self.tindex, :, self.jmin:self.jmax+1, self.imin:self.imax+1]
            vface = nc.variables['v_xy'][self.tindex, :, self.jmin:self.jmax+1, self.imin:self.imax+1]
            wface = nc.variables['w_xy'][self.tindex, :, self.jmin:self.jmax+1, self.imin:self.imax+1]
            # replace nans with zeros
            uface = np.nan_to_num(uface)
            vface = np.nan_to_num(vface)
            wface = np.nan_to_num(wface)

            # total number of grid points
            n = len(xflat)

            def velocity_fun(pos: np.ndarray) -> np.ndarray:
                """
                Velocity interpolated at pos
                pos: length 3*n vector [x..., y..., z...]
                """
                xi = pos[0:n]
                yi = pos[n:2*n]
                zi = pos[2*n:3*n]
                ifloat = (xi - xmin) / dx
                jfloat = (yi - ymin) / dy
                kfloat = (zi - zmin) / dz
                # clamp (so particle leaving domain will be evaluated at boundary cell)
                ifloat = np.clip(ifloat, 0.0, nx1 - 1.0)
                jfloat = np.clip(jfloat, 0.0, ny1 - 1.0)
                kfloat = np.clip(kfloat, 0.0, nz1 - 1.0)
                # make sure i0, j0 and k0 are on the low side
                i0 = np.clip(np.floor(ifloat).astype(np.int64), 0, nx1 - 2)
                j0 = np.clip(np.floor(jfloat).astype(np.int64), 0, ny1 - 2)
                k0 = np.clip(np.floor(kfloat).astype(np.int64), 0, nz1 - 2)

                # parametric coordinates of the cell: 0 <= xsi, eta, zet <= 1
                xsi = ifloat - i0
                eta = jfloat - j0
                zet = kfloat - k0

                isx = 1.0 - xsi
                ate = 1.0 - eta
                tez = 1.0 - zet

                # Arakawa C mimetic interpolation vector field
                # The interpolation is piecewise linear in the direction of the component 
                # and peicewise constant in the other directions. This interpolation
                # conserves fluxes and allows one to have obstacles in the domain (e.g 
                # buildings)

                # ui: linear in x between i0 and i0+1 at the same (k0,j0)
                ui = uface[k0, j0, i0] * isx + uface[k0, j0, i0 + 1] * xsi
                # vi: linear in y between j0 and j0+1 at same (k0,i0)
                vi = vface[k0, j0, i0] * ate + vface[k0, j0 + 1, i0] * eta
                # wi: linear in z between k0 and k0+1 at same (j0,i0)
                wi = wface[k0, j0, i0] * tez + wface[k0 + 1, j0, i0] * zet

                return np.concatenate([ui, vi, wi])          

            # integrate the trajectories. y is the state, a concatenated array of 
            # [x..., y..., z...] positions. y0 is the initial condition
            y0 = np.concatenate([xflat, yflat, zflat]).astype(np.float64)
            nsteps = _estimate_nsteps(uface, vface, wface, dx, dy, dz, self.tintegr)

            # Runge-Kutta 4
            y = y0.copy()
            dt = self.tintegr / nsteps
            k1 = np.empty_like(y)
            k2 = np.empty_like(y)
            k3 = np.empty_like(y)
            k4 = np.empty_like(y)
            tmp = np.empty_like(y)

            for step in range(nsteps):
                k1[:] = velocity_fun(y)
                tmp[:] = y + 0.5 * dt * k1
                k2[:] = velocity_fun(tmp)
                tmp[:] = y + 0.5 * dt * k2
                k3[:] = velocity_fun(tmp)
                tmp[:] = y + dt * k3
                k4[:] = velocity_fun(tmp)
                y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            
            final_pos = y

            # reshape
            Xf = final_pos[0:n].reshape((nz1, ny1, nx1))
            Yf = final_pos[n:2*n].reshape((nz1, ny1, nx1))
            Zf = final_pos[2*n:3*n].reshape((nz1, ny1, nx1))

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

            # build 3x3 matrices and compute max eigenvalue per cell
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

            ftle = np.log(max_lambda) / (2.0 * abs(float(self.tintegr)))
        
            return dict(
                xmin=xmin, ymin=ymin, zmin=zmin,
                dx=dx, dy=dy, dz=dz,
                nx1=nx1, ny1=ny1, nz1=nz1,
                ftle=ftle,
        )
