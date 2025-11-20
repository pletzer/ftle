import numpy as np
import scipy
import defopt

"""
Compute the Finite-Time Lyapunov Exponent (FTLE) using finite differences
to approximate the velocity gradient tensor. This involves integrating only the
trajectories and estimating the gradients via small perturbations.
"""

def compute_ftle(x, y, T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun, 
                 h=0.01, atol=1e-6, rtol=1e-6, method='LSODA'):

    # time step
    dt = T / nsteps
    n = len(x)

    def vel_fun(t, pos):
        x, y = pos[:n], pos[n:]
        u, v = u_fun(x, y), v_fun(x, y)
        return np.concatenate([u, v])

    # integrate the trajectories
    t = 0.0
    xy = np.concatenate([x, y]) # flat array of initial positions
    for _ in range(nsteps):
        result = scipy.integrate.solve_ivp(vel_fun,
                            t_span=[t, t + dt],
                            y0=xy,
                            method=method,
                            atol=atol, rtol=rtol)
        xy = result.y[:, -1] # get last position
    xf0, yf0 = xy[:n], xy[n:]

    t = 0.0
    xy = np.concatenate([x + h, y]) # flat array of initial positions
    for _ in range(nsteps):
        result = scipy.integrate.solve_ivp(vel_fun,
                            t_span=[t, t + dt],
                            y0=xy,
                            method=method,
                            atol=atol, rtol=rtol)
        xy = result.y[:, -1] # get last position
    xf1, yf1 = xy[:n], xy[n:]


    t = 0.0
    xy = np.concatenate([x, y + h]) # flat array of initial positions
    for _ in range(nsteps):
        result = scipy.integrate.solve_ivp(vel_fun,
                            t_span=[t, t + dt],
                            y0=xy,
                            method=method,
                            atol=atol, rtol=rtol)
        xy = result.y[:, -1] # get last position
    xf2, yf2 = xy[:n], xy[n:]

    # compute the velocity gradient tensor
    f11 = (xf1 - xf0) / h
    f12 = (xf2 - xf0) / h
    f21 = (yf1 - yf0) / h
    f22 = (yf2 - yf0) / h

    # compute the Cauchy-Green deformation tensor
    C11 = f11**2 + f21**2
    C12 = f11*f12 + f21*f22
    C21 = C12
    C22 = f12**2 + f22**2

    # compute the largest eigenvalue of C
    trace = C11 + C22
    det = C11*C22 - C12*C21
    lambda1 = trace / 2 + np.sqrt((trace / 2)**2 - det)

   # compute the FTLE
    ftle = np.log(lambda1) / (2 * np.abs(T))

    # det F
    detF = f11 * f22 - f12 * f21

    return {'ftle': ftle, 'detF': detF}


def test1():
    # Example usage with cateye flow
    from cateye import u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun

    # Define grid
    x = np.linspace(1.01, 1.01, 1)
    y = np.linspace(0., 0., 1)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    T = 2.0 # 10.0
    nsteps = 2 # 10
    res = compute_ftle(X.reshape(-1), Y.reshape(-1), T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun)
    ftle = res['ftle']
    detF = res['detF']
    print(f'X = {X} Y = {Y} ftle = {ftle} detF = {detF}')

def test2():
    # Example usage with cateye flow
    from cateye import u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun
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

def main(*, nx: int =100, ny: int =100, T: float =5.0, nsteps: int =10, h: float =0.01,
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
    from cateye import u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun
    import matplotlib.pyplot as plt 

    # Define grid
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    res = compute_ftle(X.reshape(-1), Y.reshape(-1), T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun,
                        h=h, atol=1e-8, rtol=1e-8, method=solver)
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

        fig.suptitle('FTLE Field for Cateye Flow using exact velocity and and finite difference gradients')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    defopt.run(main)
