import numpy as np
import scipy

"""
Compute the Finite-Time Lyapunov Exponent (FTLE) by integrating the 
velocity Jacobian along trajectories.
"""

def compute_ftle(x, y, T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun, 
                 h=0.01, atol=1e-6, rtol=1e-6, method='LSODA'):
    

    

    # time step
    dt = T / nsteps
    n = len(x)

    def tendency(t, state):

        # extract the state variables
        x, y, f11, f12, f21, f22 = state[:n], state[n:2*n], state[2*n:3*n], state[3*n:4*n], state[4*n:5*n], state[5*n:]

        xdot = u_fun(x, y)
        ydot = v_fun(x, y)

        dudx = dudx_fun(x, y)
        dudy = dudy_fun(x, y)
        dvdx = dvdx_fun(x, y)
        dvdy = dvdy_fun(x, y)

        f11dot = dudx*f11 + dudy*f21
        f12dot = dudx*f12 + dudy*f22
        f21dot = dvdx*f11 + dvdy*f21
        f22dot = dvdx*f12 + dvdy*f22

        res = np.concatenate([xdot, ydot, f11dot, f12dot, f21dot, f22dot])
        #print(xdot, ydot, f11dot, f12dot, f21dot, f22dot)
        return res

    # integrate the trajectories and the velocity Jacobian
    t = 0.0
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
    return np.log(lambda1) / (2 * np.abs(T))

def test1():
    # Example usage with cateye flow
    from cateye import u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun

    # Define grid
    x = np.linspace(1.01, 1.01, 1)
    y = np.linspace(0.0, 0.0, 1)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    T = 10.0
    nsteps = 10
    ftle = compute_ftle(X.reshape(-1), Y.reshape(-1), T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun)

    print(f'X = {X} Y = {Y} ftle = {ftle}')

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
    ftle = compute_ftle(X.reshape(-1), Y.reshape(-1), T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun,
                        h=0.01, atol=1e-8, rtol=1e-8, method='RK45')
    ftle = ftle.reshape((ny, nx))

    print(f'ftle min = {ftle.min()} max = {ftle.max()}')

    plt.pcolor(X, Y, ftle)
    plt.colorbar(label='FTLE')
    plt.title('FTLE Field for Cateye Flow using exact velocity and gradients')
    plt.show()

if __name__ == "__main__":
    test2()