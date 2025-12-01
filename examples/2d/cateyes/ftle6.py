import numpy as np
import scipy
import defopt
from ftle.deformation_gradient import getF2d

"""
Compute the Finite-Time Lyapunov Exponent (FTLE) using finite differences
to approximate the velocity gradient tensor. This involves integrating only the
trajectories and estimating the gradients via small perturbations. In this example,
the perturbation is equal to the cell size.
"""

def compute_ftle(X, Y, T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun, 
                 atol=1e-6, rtol=1e-6, method='LSODA'):
    
    ny, nx = X.shape # number of grid points in y and x directions

    # compute the staggered u, v arrays
    # Arakawa C grid
    Uface = np.empty((ny - 1, nx), float)
    Vface = np.empty((ny, nx - 1), float)

    Xu = 0.5 * (X[:-1, :] + X[1:, :])
    Yu = 0.5 * (Y[:-1, :] + Y[1:, :])
    Xv = 0.5 * (X[:, :-1] + X[:, 1:])
    Yv = 0.5 * (Y[:, :-1] + Y[:, 1:])
    Uface[:, :] = u_fun(Xu, Yu)
    Vface[:, :] = v_fun(Xv, Yv)

    # save the grid node, initial positions
    X0 = X.copy()
    Y0 = Y.copy()

    # assume uniform grid spacing
    dx = X0[0, 1] - X0[0, 0]
    dy = Y0[1, 0] - Y0[0, 0]

    # time step
    dt = T / nsteps

    # flatten the grid coordinates
    xflat = X.reshape(-1)
    yflat = Y.reshape(-1)

    # note: assume X, Y are meshgrid outputs, size is ny, nx
    ny, nx = X.shape
    n = len(xflat) # total number of points

    def vel_fun(t, pos):

        # tendency function. Array pos stores the coordinates as
        # x0, x1, ..., xn-1, y0, y1, ..., yn-1
        x, y = pos[:n], pos[n:]

        #
        # compute the velocity along the integrated points
        #

        iu0 = np.clip( np.floor( (x - Xu[0,0])/dx).astype(int), 0, nx - 1 )
        ju0 = np.clip( np.floor( (y - Yu[0,0])/dy).astype(int), 0, ny - 2 )
        iv0 = np.clip( np.floor( (x - Xv[0,0])/dx).astype(int), 0, nx - 2 )
        jv0 = np.clip( np.floor( (y - Yv[0,0])/dy).astype(int), 0, ny - 1 )

        iu1 = np.clip( iu0 + 1, 1, nx - 1 )
        ju1 = np.clip( ju0 + 1, 1, ny - 2 )
        iv1 = np.clip( iv0 + 1, 1, nx - 2 )
        jv1 = np.clip( jv0 + 1, 1, ny - 1 )

        xsiu = np.clip( (x - Xu[0,0])/dx - iu0, 0., 1. )
        etau = np.clip( (y - Yu[0,0])/dy - ju0, 0., 1. )
        xsiv = np.clip( (x - Xv[0,0])/dx - iv0, 0., 1. )
        etav = np.clip( (y - Yv[0,0])/dy - jv0, 0., 1. )

        isxu = 1.0 - xsiu
        ateu = 1.0 - etau
        isxv = 1.0 - xsiv
        atev = 1.0 - etav

        # bilinear interpolation
        u = ateu*isxu*Uface[ju0, iu0] + \
            ateu*xsiu*Uface[ju0, iu1] + \
            etau*xsiu*Uface[ju1, iu1] + \
            etau*isxu*Uface[ju1, iu0]
        
        v = atev*isxv*Vface[jv0, iv0] + \
            atev*xsiv*Vface[jv0, iv1] + \
            etav*xsiv*Vface[jv1, iv1] + \
            etav*isxv*Vface[jv1, iv0]


        return np.concatenate([u, v])

    # integrate the trajectories
    t = 0.0
    xy = np.concatenate([xflat, yflat]) # flat array of initial positions
    # step through time
    for istep in range(nsteps):
        result = scipy.integrate.solve_ivp(vel_fun,
                            t_span=[istep*t, istep*t + dt],
                            y0=xy,
                            method=method,
                            atol=atol, rtol=rtol)
        xy = result.y[:, -1] # get last position
    
    # final positions for the advected coordinate points
    Xf, Yf = xy[:n].reshape((ny, nx)), xy[n:].reshape((ny, nx))

    ((f11, f12), (f21, f22)) = getF2d(dx, dy, Xf, Yf)

    # compute the Cauchy-Green deformation tensor
    C11 = f11**2 + f21**2
    C12 = f11*f12 + f21*f22
    C21 = C12
    C22 = f12**2 + f22**2

    # compute the largest eigenvalue of C, 2D formula
    trace = C11 + C22
    det = C11*C22 - C12*C21
    lambda1 = trace / 2 + np.sqrt((trace / 2)**2 - det)

    # compute the FTLE
    ftle = np.log(lambda1) / (2 * np.abs(T))

    # det F
    detF = f11 * f22 - f12 * f21

    return {'ftle': ftle, 'detF': detF}


def test2():
    # Example usage with cateye flow
    from ftle.cateye import u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun
    import matplotlib.pyplot as plt 

    # Define grid
    nx = 100
    ny = 100
    x = np.linspace(-2.0, 2.0, nx)
    y = np.linspace(-1.5, 1.5, ny)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    T = 5.0
    nsteps = 10
    res = compute_ftle(X.reshape(-1), Y.reshape(-1), T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun,
                        h=0.001, atol=1e-8, rtol=1e-8, method='RK45') #LSODA')
    ftle = res['ftle'].reshape((ny, nx))
    detF = res['detF'].reshape((ny, nx))

    print(f'ftle min = {ftle.min()} max = {ftle.max()}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.pcolor(X, Y, ftle)
    fig.colorbar(im1, ax=ax1, label='FTLE')
    ax1.set_title('FTLE')

    im2 = ax2.pcolor(X, Y, detF-1)
    ax2.set_title('det F - 1')
    fig.colorbar(im2, ax=ax2, label='det F - 1')

    fig.suptitle('FTLE Field for Cateye Flow using exact velocity and finite difference gradients')
    plt.tight_layout()
    plt.show()
    ftle = ftle.reshape((ny, nx))

def main(*, nx: int =100, ny: int =100, T: float =5.0, nsteps: int =10,
         xmin: float =-2.0, xmax: float =2.0, ymin: float =-1.5, ymax: float =1.5,
         plot: bool =True, solver: str ='RK45'):
    """
    Compute and plot the FTLE field for the cateye flow.
    Parameters:
        nx: Number of grid points in x direction.
        ny: Number of grid points in y direction.
        T: Total integration time.
        nsteps: Number of integration steps.
        h: Finite difference step size.
        xmin, xmax: x domain limits.
        ymin, ymax: y domain limits.
        solver: ODE solver to use ('RK45', 'LSODA', etc.).
        plot: Whether to plot the results (defaut yes).
    """
    # Example usage with cateye flow
    from ftle.cateye import u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun
    import matplotlib.pyplot as plt 

    # Define grid
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    res = compute_ftle(X, Y, T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun,
                       atol=1e-8, rtol=1e-8, method=solver)
    ftle = res['ftle'].reshape((ny, nx))
    detF = res['detF'].reshape((ny, nx))

    print(f'ftle min = {ftle.min()} max = {ftle.max()}')

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        im1 = ax1.pcolor(X, Y, ftle)
        fig.colorbar(im1, ax=ax1, label='FTLE')
        ax1.set_title('FTLE')

        im2 = ax2.pcolor(X, Y, detF-1)
        ax2.set_title('det F - 1')
        fig.colorbar(im2, ax=ax2, label='det F - 1')

        fig.suptitle(f'FTLE Field for Cateye Flow using bilinear velocity and and finite difference gradients T={T}')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    defopt.run(main)
