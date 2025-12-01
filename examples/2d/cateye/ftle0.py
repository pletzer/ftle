import numpy as np
import scipy
import defopt

"""
Compute the Finite-Time Lyapunov Exponent (FTLE) by integrating the 
velocity Jacobian along with the trajectories.
"""

def compute_ftle(x, y, T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun, 
                 atol=1e-6, rtol=1e-6, method='LSODA'):
    

    

    # time step
    dt = T / nsteps
    n = len(x)

    x0 = x.copy()
    y0 = y.copy()

    def tendency(t, state):

        # extract the state variables
        x, y, f11, f12, f21, f22 = state[:n], state[n:2*n], state[2*n:3*n], state[3*n:4*n], state[4*n:5*n], state[5*n:]

        xdot = u_fun(x, y)
        ydot = v_fun(x, y)

        dudx = dudx_fun(x, y)
        dudy = dudy_fun(x, y)
        dvdx = dvdx_fun(x, y)
        dvdy = dvdy_fun(x, y)

        # F is the velocity Jacobian F = (grad u)
        f11dot = dudx*f11 + dudy*f21
        f12dot = dudx*f12 + dudy*f22
        f21dot = dvdx*f11 + dvdy*f21
        f22dot = dvdx*f12 + dvdy*f22

        res = np.concatenate([xdot, ydot, f11dot, f12dot, f21dot, f22dot])
        #print(xdot, ydot, f11dot, f12dot, f21dot, f22dot)
        return res

    # integrate the trajectories and the velocity Jacobian
    t = 0.0
    # initially F is identity
    f11 = np.ones(n)
    f12 = np.zeros(n)
    f21 = np.zeros(n)
    f22 = np.ones(n)
    state = np.concatenate([x, y, f11, f12, f21, f22]) 
    for _ in range(nsteps):
        result = scipy.integrate.solve_ivp(tendency,
                            t_span=[t, t + dt],
                            y0=state,
                            method=method,
                            atol=atol, rtol=rtol)
        state = result.y[:, -1] # get last state

    f11 = state[2*n:3*n]
    f12 = state[3*n:4*n]
    f21 = state[4*n:5*n]
    f22 = state[5*n:]

    xf = state[:n]
    yf = state[n:2*n]
   
    # compute the Cauchy-Green deformation tensor
    # C = F^T . F
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

    return {'ftle': ftle, 'detF': detF, 'x0': x0, 'y0': y0, 'xf': xf, 'yf': yf}


def test1():
    # Example usage with cateye flow
    from ftle.cateye import u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun

    # Define grid
    x = np.linspace(1.01, 1.01, 1)
    y = np.linspace(0.0, 0.0, 1)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    T = 2.0 # 10.0
    nsteps = 2 # 10
    res = compute_ftle(X.reshape(-1), Y.reshape(-1), T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun)

    print(f'X = {X} Y = {Y} ftle = {res["ftle"]} detF = {res["detF"]}')

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
                        atol=1e-8, rtol=1e-8, method='RK45') # LSODA struggles here
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

    fig.suptitle('FTLE Field for Cateye Flow using exact velocity and gradients')
    plt.tight_layout()
    plt.show()

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
        xmin, xmax: x domain limits.
        ymin, ymax: y domain limits.
        plot: Whether to plot the results (defaut yes)
        solver: ODE solver to use ('RK45', 'LSODA', etc.).
    """
    # Example usage with cateye flow
    from ftle.cateye import u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun
    import matplotlib.pyplot as plt 

    # Define grid
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    res = compute_ftle(X.reshape(-1), Y.reshape(-1), T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun,
                        atol=1e-8, rtol=1e-8, method=solver) # LSODA struggles here
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

        # show the start/end points
        plt.quiver(
            res['x0'], res['y0'],     # tail positions
            res['xf'] - res['x0'], res['yf'] - res['y0'],       # arrow components
        angles='xy', scale_units='xy') #, scale=scale


        fig.suptitle(f'FTLE Field for Cateye Flow using exact velocity and gradients T={T}')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    defopt.run(main)
