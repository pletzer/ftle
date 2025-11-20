from numpy import pi, sin, cos
import numpy as np
from numba import cfunc, njit
from numbalsoda import lsoda_sig

def stream_fun(x, y):
    return y**2/2 + (sp.sin(sp.pi*x/2))**2 / 2

def u_fun(x, y):
    return y

def v_fun(x, y):
    return -pi*sin(pi*x/2)*cos(pi*x/2)/2

def dudx_fun(x, y):
    return np.zeros_like(x)

def dudy_fun(x, y):
    return np.ones_like(y)

def dvdx_fun(x, y):
    return pi**2*sin(pi*x/2)**2/4 - pi**2*cos(pi*x/2)**2/4

def dvdy_fun(x, y):
    return np.zeros_like(x)

# for numbaCS
@cfunc(lsoda_sig)
def cateye_flow(t, xy, uv, param):
    x = xy[0]
    y = xy[1]
    uv[0] = y
    uv[1] = -pi*sin(pi*x/2)*cos(pi*x/2)/2

