import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m, k, c = 2.0, 10.0, 0.1
x0, v0 = 1.0, 0.0
dt, T = 0.1, 20.0
N = int(round(T / dt))
a, b = k / m, c / m
D = 1 + dt * b + dt**2 * a

# ---- Simulation: step the state with a linear solve, then compute E and difference it ----
M_ = np.array([[1.0, -dt], [dt * a, 1.0 + dt * b]])
x = np.zeros(N + 1); v = np.zeros(N + 1)
x[0], v[0] = x0, v0
for n in range(N):
    x[n + 1], v[n + 1] = np.linalg.solve(M_, [x[n], v[n]])
E = 0.5 * m * v**2 + 0.5 * k * x**2
dE_sim = np.diff(E)

# ---- Derived formula: dE = -dt*c*v'^2 - m/2*(v'-v)^2 - k/2*(x'-x)^2 ----
# v', x' come from the substitution formulas, using only the OLD state (x_n, v_n)
xn, vn = x[:-1], v[:-1]
v_new = (vn - dt * a * xn) / D
x_new = (xn * (1 + dt * b) + dt * vn) / D
dE_formula = (-dt * c * v_new**2
              - 0.5 * m * (v_new - vn)**2
              - 0.5 * k * (x_new - xn)**2)

print("max |sim - formula| =", np.max(np.abs(dE_sim - dE_formula)))

t = np.arange(N) * dt
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, dE_sim, "b", lw=4, alpha=0.5, label="simulated: diff of E each step")
ax.plot(t, dE_formula, "k--", lw=1.5, label="derived implicit-Euler ΔE formula")
ax.axhline(0, color="gray", lw=0.5)
ax.set(xlabel="t [s]", ylabel="ΔE per step [J]",
       title=f"Implicit Euler ΔE: derived vs simulated (m={m}, k={k}, c={c}, dt={dt})")
ax.legend()
fig.tight_layout()
fig.savefig("implicit_derived_vs_sim_dE.png", dpi=140)