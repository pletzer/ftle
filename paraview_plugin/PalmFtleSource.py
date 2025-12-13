"""
Custom ParaView Python Source plugin to read and compute the Finite Time Lyapunov Exponent 
from PALM data

Inputs:
  - palmfile: path to a NetCDF file
  - tintegr: integration time (float)
  - imin, imax: x-index bounds
  - jmin, jmax: y-index bounds

Reads fields:
  - u_xy, v_xy, w_xy

Grid:
  - Assumed uniform, 3D, cell-centred output
  - Index order assumed (k, j, i) = (nz, ny, nx)
"""

from paraview.util.vtkAlgorithm import *
import numpy as np
import netCDF4
import vtk


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
        self.imax = 0
        self.jmin = 0
        self.jmax = 0

    # ------------------------------------------------------------------
    # Properties exposed to ParaView GUI
    # ------------------------------------------------------------------

    @smproperty.stringvector(name="PalmFile", number_of_elements=1)
    @smdomain.filelist()
    def SetPalmFile(self, value):
        self.palmfile = value[0]
        self.Modified()

    @smproperty.doublevector(name="IntegrationTime", number_of_elements=1, default_values=[1.0])
    def SetIntegrationTime(self, value, *args):
        self.tintegr = float(value)
        self.Modified()

    @smproperty.intvector(name="IRange", number_of_elements=2, default_values=[0, -1])
    def SetIBounds(self, value, *args):
        if isinstance(value, (list, tuple)):
            self.imin, self.imax = map(int, value)
        else:
            # single value
            self.imin = int(value)
            self.imax = self.imin + 2
        self.Modified()

    @smproperty.intvector(name="JRange", number_of_elements=2, default_values=[0, -1])
    def SetJBounds(self, value, *args):
        if isinstance(value, (list, tuple)):
            self.jmin, self.jmax = map(int, value)
        else:
            # single value
            self.jmin = int(value)
            self.jmax = self.imin + 2
        self.Modified()

    # ------------------------------------------------------------------
    # Core pipeline method
    # ------------------------------------------------------------------

    def RequestData(self, request, inInfo, outInfo):
        if not self.palmfile:
            raise RuntimeError("PalmFile must be specified")

        # --------------------------------------------------------------
        # Read NetCDF data
        # --------------------------------------------------------------
        with netCDF4.Dataset(self.palmfile, "r") as nc:
            # Expect shape (nz, ny, nx)
            u = nc.variables['u_xy'][:]
            v = nc.variables['v_xy'][:]
            w = nc.variables['w_xy'][:]

            # Optional: coordinates if present
            x = nc.variables.get('x')
            y = nc.variables.get('y')
            z = nc.variables.get('z')

        # Arakawa C 
        nz1, ny, nx = u.shape

        # --------------------------------------------------------------
        # Subset in i, j (cell indices)
        # --------------------------------------------------------------
        i0, i1 = self.imin, self.imax
        j0, j1 = self.jmin, self.jmax

        if i1 <= i0 + 2 or j1 <= j0 + 2:
            raise RuntimeError("Invalid (imin,imax,jmin,jmax) bounds")

        u_sub = u[:, j0:j1, i0:i1]
        v_sub = v[:, j0:j1, i0:i1]
        w_sub = w[:, j0:j1, i0:i1]

        nzs, nys, nxs = u_sub.shape

        # --------------------------------------------------------------
        # Placeholder for FTLE computation
        # --------------------------------------------------------------
        # Allocate output (cell-centred FTLE)
        ftle = np.zeros((nzs, nys, nxs), dtype=np.float64)

        # --------------------------------------------------------------
        # INSERT YOUR FTLE COMPUTATION HERE
        #
        # You have access to:
        #   u_sub, v_sub, w_sub   : velocities
        #   self.tintegr          : integration time
        #   index order           : (k, j, i)
        #
        # Expected result:
        #   ftle[k, j, i]
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # Convert NumPy array to vtkImageData
        # --------------------------------------------------------------
        output = vtk.vtkImageData.GetData(outInfo, 0)

        output.SetDimensions(nxs, nys, nzs)
        output.SetOrigin(0.0, 0.0, 0.0)
        output.SetSpacing(1.0, 1.0, 1.0)  # uniform grid assumed

        vtk_array = vtk.util.numpy_support.numpy_to_vtk(
            num_array=ftle.ravel(order='F'),
            deep=True,
            array_type=vtk.VTK_DOUBLE
        )
        vtk_array.SetName("FTLE")

        output.GetCellData().AddArray(vtk_array)
        output.GetCellData().SetScalars(vtk_array)

        return 1


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
