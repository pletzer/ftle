import numpy as np

def getFTLE2d(hx: float, hy: float, Xf: np.ndarray, Yf: np.ndarray, T: float):
    """
    Get the finite time Lyapunov exponent 
    @param hx grid cell size in x
    @param hy grid cell size in y 
    @param Xf final x coordinates
    @param Yf final y coordinates
    @param T time
    @return tensor ((f11, f12), (f21, f22))
    """
    
    # allocate deformation gradient
    f11 = np.empty_like(Xf)
    f12 = np.empty_like(Xf)
    f21 = np.empty_like(Xf)
    f22 = np.empty_like(Xf)

    # dX/dx (ny, nx)
    f11[:, 1:-1] = (Xf[:,2:] - Xf[:,:-2]) / (2*hx)
    f11[:,  0] = (Xf[:, 1] - Xf[:, 0]) / hx # one sided differences
    f11[:, -1] = (Xf[:,-1] - Xf[:,-2]) / hx

    # dX/dy
    f12[1:-1, :] = (Xf[2:,:] - Xf[:-2,:]) / (2*hy)
    f12[ 0, :] = (Xf[ 1,:] - Xf[ 0,:]) / hy # one sided differences
    f12[-1, :] = (Xf[-1,:] - Xf[-2,:]) / hy

    # dY/dx
    f21[:, 1:-1] = (Yf[:,2:] - Yf[:,:-2]) / (2*hx)
    f21[:,  0] = (Yf[:, 1] - Yf[:, 0]) / hx # one sided differences
    f21[:, -1] = (Yf[:,-1] - Yf[:,-2]) / hx

    # dY/dy
    f22[1:-1, :] = (Yf[2:,:] - Yf[:-2,:]) / (2*hy)
    f22[ 0, :] = (Yf[ 1,:] - Yf[ 0,:]) / hy # one sided differences
    f22[-1, :] = (Yf[-1,:] - Yf[-2,:]) / hy

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
   