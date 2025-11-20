import numpy as np

def test2():
    # Example usage with cateye flow
    from cateye import cateye_flow
    from numbacs.integration import flowmap_grid_2D
    from numbacs.diagnostics import ftle_grid_2D
    import matplotlib.pyplot as plt 

    # Define grid
    nx = 100
    ny = 100
    x = np.linspace(-2.0, 2.0, nx)
    y = np.linspace(-1.5, 1.5, ny)
    X, Y = np.meshgrid(x, y)

    # Compute FTLE
    T = 5.0
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    params = np.array([1.0], dtype=np.float64) 
    flowmap = flowmap_grid_2D(cateye_flow.address, t0=0., T=T, x=x, y=y, params=params, method='lsoda', rtol=1e-8, atol=1e-8)
    ftle = ftle_grid_2D(flowmap, T, dx, dy)

    print(f'ftle min = {ftle.min()} max = {ftle.max()}')

    plt.pcolor(x, y, ftle.T)
    plt.colorbar(label='FTLE')
    plt.title('FTLE for Cateye Flow using numbaCS')
    plt.show()


if __name__ == "__main__":
    test2()