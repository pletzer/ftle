"""
flow around a circle
"""

import numpy as np
import defopt
import matplotlib.pyplot as plt

def uv_fun(x, y, a: float=1.0):
    z = x + 1j*y
    w = 1.0 + a**2/z**2
    u = w.real
    v = -w.imag # complex conjugate
    return u, v

def main(*, L: float=10, H: float=5.0, nx: int=20, ny: int=10, a: float=2.0):
    x = np.linspace(-L/2.0, L/2.0, nx)
    y = np.linspace(0., H, ny)
    uu = np.zeros((ny, nx), float)
    vv = np.zeros((ny, nx), float)
    for j in range(ny):
        for i in range(nx):
            inside = (x[i]**2 + y[j]**2 < a**2)
            if not inside:
                uu[j, i], vv[j, i] = uv_fun(x[i], y[j], a=a)

    # plot
    plt.quiver(x, y, uu, vv)
    plt.streamplot(x, y, uu, vv)
    th = np.linspace(0.0, np.pi, 128)
    xc = a*np.cos(th)
    yc = a*np.sin(th)
    plt.plot(xc, yc)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    defopt.run(main)

