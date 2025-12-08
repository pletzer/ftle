
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -----------------------------
# Geometry & parameters
# -----------------------------
L, H = 10.0, 5.0
x1, y1, x2, y2 = 2.0, 1.0, 3.0, 3.0
U = 1.0                  # sets scale: phi(L,y) = L*U
Nx, Ny = 41, 21        # try 201x101; increase if you want finer
x = np.linspace(0, L, Nx)
y = np.linspace(0, H, Ny)
dx, dy = x[1]-x[0], y[1]-y[0]
X, Y = np.meshgrid(x, y, indexing='ij')

# Fluid mask (True=fluid, False=solid)
fluid = np.ones((Nx, Ny), dtype=bool)
fluid[(X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)] = False

# -----------------------------
# Potential solve: Laplace via SOR
# -----------------------------
phi = np.zeros((Nx, Ny))
phi[0, :]  = 0.0
phi[-1, :] = L * U
for i in range(1, Nx-1):           # linear initial guess
    phi[i, :] = (x[i]/L) * (L*U)

omega    = 1.9                      # SOR relaxation
max_iter = 6000
tol      = 1e-6

def apply_wall_neumann(arr):
    # dphi/dy = 0 at walls -> copy from interior
    arr[:, 0]  = arr[:, 1]
    arr[:, -1] = arr[:, -2]

for it in range(max_iter):
    phi_old = phi.copy()
    apply_wall_neumann(phi)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            if not fluid[i, j]:
                continue
            # mirrored neighbors if across solid (enforces internal Neumann)
            pim1 = phi[i-1, j] if fluid[i-1, j] else phi[i, j]
            pip1 = phi[i+1, j] if fluid[i+1, j] else phi[i, j]
            pjm1 = phi[i, j-1] if fluid[i, j-1] else phi[i, j]
            pjp1 = phi[i, j+1] if fluid[i, j+1] else phi[i, j]
            phi_avg = 0.25*(pim1 + pip1 + pjm1 + pjp1)
            phi[i, j] = (1-omega)*phi[i, j] + omega*phi_avg
    phi[0, :]  = 0.0
    phi[-1, :] = L * U
    apply_wall_neumann(phi)
    if np.linalg.norm(phi - phi_old, ord=np.inf) < tol:
        print(f"Converged in {it} iterations")
        break
else:
    print("Reached max_iter; consider increasing iterations or adjusting omega.")

# -----------------------------
# Velocities with proper solid Neumann handling
# -----------------------------
u = np.zeros_like(phi)
v = np.zeros_like(phi)
for i in range(1, Nx-1):
    for j in range(1, Ny-1):
        if not fluid[i, j]:
            continue
        pim1 = phi[i-1, j] if fluid[i-1, j] else phi[i, j]
        pip1 = phi[i+1, j] if fluid[i+1, j] else phi[i, j]
        pjm1 = phi[i, j-1] if fluid[i, j-1] else phi[i, j]
        pjp1 = phi[i, j+1] if fluid[i, j+1] else phi[i, j]
        u[i, j] = (pip1 - pim1)/(2*dx)
        v[i, j] = (pjp1 - pjm1)/(2*dy)

# One-sided at outer boundaries (kept simple)
u[0, :]  = (phi[1, :] - phi[0, :]) / dx
u[-1, :] = (phi[-1, :] - phi[-2, :]) / dx
v[:, 0]  = (phi[:, 1] - phi[:, 0]) / dy
v[:, -1] = (phi[:, -1] - phi[:, -2]) / dy

# -----------------------------
# Diagnostics
# -----------------------------
lap = np.zeros_like(phi)
for i in range(1, Nx-1):
    for j in range(1, Ny-1):
        if not fluid[i, j]: continue
        pim1 = phi[i-1, j] if fluid[i-1, j] else phi[i, j]
        pip1 = phi[i+1, j] if fluid[i+1, j] else phi[i, j]
        pjm1 = phi[i, j-1] if fluid[i, j-1] else phi[i, j]
        pjp1 = phi[i, j+1] if fluid[i, j+1] else phi[i, j]
        lap[i, j] = (pip1 - 2*phi[i, j] + pim1)/dx**2 + (pjp1 - 2*phi[i, j] + pjm1)/dy**2

curl = np.zeros_like(phi)
curl[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1])/(2*dx) - (u[1:-1, 2:] - u[1:-1, :-2])/(2*dy)

fluid_interior = fluid.copy()
fluid_interior[[0, -1], :] = False
fluid_interior[:, [0, -1]] = False

print("mean |Δφ| (fluid)  :", np.nanmean(np.abs(lap[fluid_interior])))
print("mean |curl u|      :", np.nanmean(np.abs(curl[fluid_interior])))

# Flux constancy Q(x) = ∫ u(x,y) dy over fluid segments
Q = []
for i in range(2, Nx-2):
    mask_y = fluid[i, :]
    u_line = u[i, :].copy()
    u_line[~mask_y] = np.nan
    q_i, j = 0.0, 0
    while j < Ny:
        if not mask_y[j]: j += 1; continue
        s = j
        while j < Ny and mask_y[j]: j += 1
        e = j
        q_i += np.trapz(u_line[s:e], y[s:e])
    Q.append(q_i)
print("Q mean, std, min, max:", np.nanmean(Q), np.nanstd(Q), np.nanmin(Q), np.nanmax(Q))

# -----------------------------
# Plotting
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
c = ax.contourf(X, Y, phi, levels=40, cmap='viridis')
ax.contour(X, Y, phi, levels=40, colors='k', linewidths=0.3)
ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, facecolor='lightgray', edgecolor='k'))
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Potential $\\phi$')
fig.colorbar(c, ax=ax, label='$\\phi$')

ax2 = axes[1]
# mask obstacle in streamlines
u_mask = np.ma.masked_where(~fluid, u)
v_mask = np.ma.masked_where(~fluid, v)
speedT = np.hypot(u_mask.T, v_mask.T)
X_plot, Y_plot = np.meshgrid(x, y)  # default 'xy'
strm = ax2.streamplot(X_plot, Y_plot, u_mask.T, v_mask.T, color=speedT,
                      cmap='plasma', density=1.6)
ax2.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, facecolor='lightgray', edgecolor='k'))
ax2.set_aspect('equal'); ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.set_title('Streamlines ($\\mathbf{u}=\\nabla\\phi$)')
plt.tight_layout()
plt.show()
