import numpy as np
import defopt
from cateye import cateye_flow
from numbacs.integration import flowmap_grid_2D
from numbacs.diagnostics import ftle_grid_2D
import matplotlib.pyplot as plt 

  
def test2():
    # Example usage with cateye flow
    # Define grid
    nx = 100
    ny = 100
    x = np.linspace(-2.0, 2.0, nx)
    y = np.linspace(-1.5, 1.5, ny)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    T = 5.0
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    params = np.array([1.0], dtype=np.float64) 
    flowmap = flowmap_grid_2D(cateye_flow.address, t0=0., T=T, x=x, y=y, params=params, method='lsoda', rtol=1e-8, atol=1e-8)
    ftle = ftle_grid_2D(flowmap, T, dx, dy)

    print(f'ftle min = {ftle.min()} max = {ftle.max()}')

    plt.pcolor(x, y, ftle.T)
    plt.colorbar(label='FTLE')
    plt.title('FTLE for Cateye Flow using numbaCS')
    plt.show()

def main(*, nx: int =100, ny: int =100, T: float =5.0, 
         xmin: float =-2.0, xmax: float =2.0, ymin: float =-1.5, ymax: float =1.5,
         plot: bool =True, solver: str ='LSODA'):
    """
    Compute and plot the FTLE field for the cateye flow.
    Parameters:
        nx: Number of grid points in x direction.
        ny: Number of grid points in y direction.
        T: Total integration time.
        nsteps: Number of integration steps.
        xmin, xmax: x domain limits.
        ymin, ymax: y domain limits.
        plot: Whether to plot the results (defaut yes).
        solver: ODE solver to use ('RK45', 'LSODA', etc.).
    """

    # Define grid
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # need this parameter to set the time
    params = np.array([1.0], dtype=np.float64) 
    flowmap = flowmap_grid_2D(cateye_flow.address, t0=0., T=T, x=x, y=y, params=params, method=solver, rtol=1e-8, atol=1e-8)
    ftle = ftle_grid_2D(flowmap, T, dx, dy)

    print(f'ftle min = {ftle.min()} max = {ftle.max()}')

    if plot:
        plt.pcolor(x, y, ftle.T)
        plt.colorbar(label='FTLE')
        plt.title('FTLE for Cateye Flow using numbaCS')
        plt.show()


if __name__ == "__main__":
    defopt.run(main)