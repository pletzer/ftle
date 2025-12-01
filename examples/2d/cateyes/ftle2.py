import numpy as np
import scipy
import defopt

def expm_2x2_trace_free(a, b, c, d):
    """
    Compute the matrix exponential of a 2x2, trace-free matrix A using the closed-form formula.
    """

    # Initialize result arrays
    r11 = np.zeros_like(a)
    r12 = np.zeros_like(a)
    r21 = np.zeros_like(a)
    r22 = np.zeros_like(a)

    delta_sq = -a*d + b*c # -det A

    mask_pos = delta_sq > 0
    mask_zero = delta_sq == 0
    mask_neg = delta_sq < 0

    # Hyperbolic case
    if np.any(mask_pos):
        delta = np.sqrt(delta_sq[mask_pos])
        cosh_delta = np.cosh(delta)
        sinh_delta = np.sinh(delta)

        r11[mask_pos] = cosh_delta * 1 + (sinh_delta / delta) * a[mask_pos]
        r12[mask_pos] = cosh_delta * 0 + (sinh_delta / delta) * b[mask_pos]
        r21[mask_pos] = cosh_delta * 0 + (sinh_delta / delta) * c[mask_pos]
        r22[mask_pos] = cosh_delta * 1 + (sinh_delta / delta) * d[mask_pos] 

    # Zero case
    if np.any(mask_zero):
        r11[mask_zero] = 1 + a[mask_zero]
        r12[mask_zero] =     b[mask_zero]
        r21[mask_zero] =     c[mask_zero]
        r22[mask_zero] = 1 + d[mask_zero]
     
    # Elliptic case
    if np.any(mask_neg):
        gamma = np.sqrt(-delta_sq[mask_neg])
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)

        r11[mask_neg] = cos_gamma * 1 + (sin_gamma / gamma) * a[mask_neg]
        r12[mask_neg] = cos_gamma * 0 + (sin_gamma / gamma) * b[mask_neg]
        r21[mask_neg] = cos_gamma * 0 + (sin_gamma / gamma) * c[mask_neg]
        r22[mask_neg] = cos_gamma * 1 + (sin_gamma / gamma) * d[mask_neg]

    return r11, r12, r21, r22


"""
Compute the Finite-Time Lyapunov Exponent (FTLE) by using a single step when 
advecting the velocity Jacobian. The step conserves volume exactly.
"""

def compute_ftle(x, y, T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun, 
                 atol=1e-6, rtol=1e-6, method='LSODA'):

    # time step
    dt = T / nsteps
    n = len(x)

    def vel_fun(t, pos):
        x, y = pos[:n], pos[n:]
        u, v = u_fun(x, y), v_fun(x, y)
        return np.concatenate([u, v])

    # integrate the trajectories (half step for velocity Jacobian)
    f11 = np.ones(n)
    f12 = np.zeros(n)
    f21 = np.zeros(n)
    f22 = np.ones(n)
    t = 0.0
    xy = np.concatenate([x, y]) # flat array of initial positions
    for _ in range(nsteps):

        # half step to get positions for velocity gradient evaluation
        result = scipy.integrate.solve_ivp(vel_fun,
                            t_span=[t, t + 0.5*dt],
                            y0=xy,
                            method=method,
                            atol=atol, rtol=rtol)
        xy = result.y[:, -1] # get last position
        t += 0.5*dt

        # extract final positions after half step
        x, y = xy[:n], xy[n:]

        # evaluate velocity gradients at half step positions
        a11dt = dudx_fun(x, y) * dt
        a12dt = dudy_fun(x, y) * dt
        a21dt = dvdx_fun(x, y) * dt
        a22dt = dvdy_fun(x, y) * dt

        # take the exponential of the velocity gradient tensor times dt
        E11, E12, E21, E22 = expm_2x2_trace_free(a11dt, a12dt, a21dt, a22dt)
        #E11, E12, E21, E22 = expm_trace_zero_elements(a11dt, a12dt, a21dt)

        # update velocity gradient tensor
        f11_new = E11 * f11 + E12 * f21
        f12_new = E11 * f12 + E12 * f22
        f21_new = E21 * f11 + E22 * f21
        f22_new = E21 * f12 + E22 * f22
        f11, f12, f21, f22 = f11_new, f12_new, f21_new, f22_new

        # finish full step to get new positions
        result = scipy.integrate.solve_ivp(vel_fun,
                            t_span=[t, t + 0.5*dt],
                            y0=xy,
                            method=method,
                            atol=atol, rtol=rtol)
        xy = result.y[:, -1] # get last position
        t += 0.5*dt

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
    from ftle.cateye import u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun

    # Define grid
    x = np.linspace(1.01, 1.01, 1)
    y = np.linspace(0., 0., 1)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    T = 5.0
    nsteps = 10
    res = compute_ftle(X.reshape(-1), Y.reshape(-1), T, nsteps, u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun)
    ftle = res['ftle']
    detF = res['detF']
    print(f'X = {X} Y = {Y} ftle = {ftle} detF = {detF}')

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
                       atol=1e-8, rtol=1e-8, method='LSODA')
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

    fig.suptitle(f'FTLE Field for Cateye Flow using exact velocity and area preserving F integration T={T}')
    plt.tight_layout()
    plt.show()
    ftle = ftle.reshape((ny, nx))

def main(*, nx: int =100, ny: int =100, T: float =5.0, nsteps: int =10, 
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

        fig.suptitle('FTLE Field for Cateye Flow using exact velocity and area preserving F integration')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    defopt.run(main)
