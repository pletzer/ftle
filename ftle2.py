import numpy as np
import scipy

import numpy as np

def expm_trace_zero_elements(a, b, c):
    """
    Compute exp(A) for many 2x2 matrices A with zero trace.
    A = [[a, b],
         [c, -a]]
    
    Handles both elliptic (rotation-like) and hyperbolic cases.
    
    Returns:
        r11, r12, r21, r22 : arrays of same shape as a
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    
    if not (a.shape == b.shape == c.shape):
        raise ValueError("a, b, c must have the same shape")
    
    disc = a**2 + b*c  # discriminant
    r11 = np.zeros_like(a, dtype=float)
    r12 = np.zeros_like(a, dtype=float)
    r21 = np.zeros_like(a, dtype=float)
    r22 = np.zeros_like(a, dtype=float)
    
    # Elliptic case: disc > 0
    mask_pos = disc > 0
    gamma = np.sqrt(disc[mask_pos])
    cos_g = np.cos(gamma)
    sin_g_over_gamma = np.sin(gamma) / gamma
    r11[mask_pos] = cos_g + sin_g_over_gamma * a[mask_pos]
    r12[mask_pos] =         sin_g_over_gamma * b[mask_pos]
    r21[mask_pos] =         sin_g_over_gamma * c[mask_pos]
    r22[mask_pos] = cos_g - sin_g_over_gamma * a[mask_pos]
    
    # Hyperbolic case: disc < 0
    mask_neg = disc < 0
    beta = np.sqrt(-disc[mask_neg])
    cosh_b = np.cosh(beta)
    sinh_b_over_beta = np.sinh(beta) / beta
    r11[mask_neg] = cosh_b + sinh_b_over_beta * a[mask_neg]
    r12[mask_neg] =          sinh_b_over_beta * b[mask_neg]
    r21[mask_neg] =          sinh_b_over_beta * c[mask_neg]
    r22[mask_neg] = cosh_b - sinh_b_over_beta * a[mask_neg]
    
    # Zero case: disc == 0
    mask_zero = disc == 0
    r11[mask_zero] = 1 + a[mask_zero]
    r12[mask_zero] =     b[mask_zero]
    r21[mask_zero] =     c[mask_zero]
    r22[mask_zero] = 1 - a[mask_zero]
    
    return r11, r12, r21, r22




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

    
    # # Compute invariants
    # delta_sq = a*a + b*c #-a*d + b*c
    # #print(f'delta_sq = {delta_sq}')
    # if delta_sq.any() < 0:
    #     RuntimeError(f'Some delta_sq are negative= {delta_sq}')

    # # delta_sq > 0
    # delta = np.sqrt(delta_sq)
            
    
    # # Compute exponential
    # cosh_delta = np.cosh(delta)
    # sinh_delta = np.sinh(delta)

    # r11 = cosh_delta * 1 + (sinh_delta / delta) * a
    # r12 = cosh_delta * 0 + (sinh_delta / delta) * b
    # r21 = cosh_delta * 0 + (sinh_delta / delta) * c
    # r22 = cosh_delta * 1 + (sinh_delta / delta) * d
    
    # return r11, r12, r21, r22



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
    from cateye import u_fun, v_fun, dudx_fun, dudy_fun, dvdx_fun, dvdy_fun

    # Define grid
    x = np.linspace(1.01, 1.01, 1)
    y = np.linspace(0., 0., 1)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    T = 2.0
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

    fig.suptitle('FTLE Field for Cateye Flow using exact velocity and finite difference grandients')
    plt.tight_layout()
    plt.show()
    ftle = ftle.reshape((ny, nx))


if __name__ == "__main__":
    test1()
    test2()