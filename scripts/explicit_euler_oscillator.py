import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m, k, c = 2.0, 10.0, 0.1    
x0, v0 = 1.0, 0.0
dt, T = 0.1, 20.0
N = int(round(T / dt))

x = np.zeros(N + 1); v = np.zeros(N + 1)
x[0], v[0] = x0, v0
for n in range(N):
    a_n = -(k / m) * x[n] - (c / m) * v[n]
    x[n + 1] = x[n] + dt * v[n]
    v[n + 1] = v[n] + dt * a_n

E = 0.5 * m * v**2 + 0.5 * k * x**2
dE = np.diff(E)
t = np.arange(N) * dt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, dE, "r", lw=2)
ax.axhline(0, color="gray", lw=0.5)
ax.set(xlabel="t [s]", ylabel="ΔE per step [J]",
       title=f"Explicit Euler: ΔE per step (m={m}, k={k}, c={c}, dt={dt})")
fig.tight_layout()
fig.savefig("explicit_euler_dE.png", dpi=140)