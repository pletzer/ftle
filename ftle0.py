import numpy as np
import scipy

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
    dxpdx = (xf1 - xf0) / h
    dxpdy = (xf2 - xf0) / h
    dypdx = (yf1 - yf0) / h
    dypdy = (yf2 - yf0) / h
    
    # compute the Cauchy-Green deformation tensor
    C11 = dxpdx**2 + dypdx**2
    C12 = dxpdx*dxpdy + dypdx*dypdy
    C21 = C12
    C22 = dxpdy**2 + dypdy**2

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
    x = np.linspace(1.0, 1.0, 1)
    y = np.linspace(0., 0., 1)
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
    nx = 80
    ny = 80
    x = np.linspace(-2.0, 2.0, nx)
    y = np.linspace(-1.5, 1.5, ny)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    T = 10.0
    nsteps = 10
    ftle = compute_ftle(X.reshape(-1), Y.reshape(-1), T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun)
    ftle = ftle.reshape((ny, nx))

    plt.pcolor(X, Y, ftle)
    plt.colorbar(label='FTLE')
    plt.title('FTLE Field for Cateye Flow using analytic velocity field')
    plt.show()

if __name__ == "__main__":
    test2()