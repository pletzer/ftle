import numpy as np
import scipy

"""
Compute the Finite-Time Lyapunov Exponent (FTLE) using finite differences
to approximate the velocity gradient tensor.
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
    ftle = np.log(lambda1) / (2 * np.abs(T))

    # det F
    detF = dxpdx * dypdy - dxpdy * dypdx

    return {'ftle': ftle, 'detF': detF}

def test1():
    # Example usage with cateye flow
    from cateye import u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun

    # Define grid
    x = np.linspace(1.01, 1.01, 1)
    y = np.linspace(0., 0., 1)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    T = 10.0
    nsteps = 10
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
                        h=0.01, atol=1e-8, rtol=1e-8, method='LSODA')
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

    fig.suptitle('FTLE Field for Cateye Flow using exact velocity and finite difference grandients')
    plt.tight_layout()
    plt.show()
    ftle = ftle.reshape((ny, nx))

    print(f'ftle min = {ftle.min()} max = {ftle.max()}')

    plt.pcolor(X, Y, ftle)
    plt.colorbar(label='FTLE')
    plt.title('FTLE Field for Cateye Flow using exact velocity and finite difference')
    plt.show()

if __name__ == "__main__":
    test1()
    test2()