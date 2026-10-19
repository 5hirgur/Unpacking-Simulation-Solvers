import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m, k, c = 2.0, 10.0, 0.1
x0, v0 = 1.0, 0.0
dt, T = 0.1, 20.0
N = int(round(T / dt))
a, b = k / m, c / m
t = np.arange(N + 1) * dt

def energy(x, v):
    return 0.5 * m * v**2 + 0.5 * k * x**2

# ---- 1) COMPUTED: step the semi-implicit scheme, difference the energy ----
x = np.zeros(N + 1); v = np.zeros(N + 1)
x[0], v[0] = x0, v0
for n in range(N):
    v[n + 1] = v[n] + dt * (-a * x[n] - b * v[n])   # v from OLD state
    x[n + 1] = x[n] + dt * v[n + 1]                 # x from NEW v
dE_computed = np.diff(energy(x, v))

# ---- 2) DERIVED: dE = -dt*c*v*v' - m/2*(v'-v)^2 + k/2*(x'-x)^2 ----
xn, vn = x[:-1], v[:-1]
v_new = vn + dt * (-a * xn - b * vn)
x_new = xn + dt * v_new
dE_derived = (-dt * c * vn * v_new
              - 0.5 * m * (v_new - vn)**2
              + 0.5 * k * (x_new - xn)**2)

# ---- 3) EXACT: closed-form damped oscillator, energy differenced at the same times ----
alpha = c / (2 * m)
wd = np.sqrt(k / m - alpha**2)
A, B = x0, (v0 + alpha * x0) / wd
x_ex = np.exp(-alpha * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))
v_ex = np.exp(-alpha * t) * ((-alpha * A + wd * B) * np.cos(wd * t)
                             + (-alpha * B - wd * A) * np.sin(wd * t))
dE_exact = np.diff(energy(x_ex, v_ex))

print("max |computed - derived| =", np.max(np.abs(dE_computed - dE_derived)))
print("E start/end: computed %.4f -> %.4f, exact %.4f -> %.4f"
      % (energy(x, v)[0], energy(x, v)[-1], energy(x_ex, v_ex)[0], energy(x_ex, v_ex)[-1]))

tt = t[:-1]
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(tt, dE_computed, color="tab:green", lw=4, alpha=0.5, label="computed (stepped, diff of E)")
ax.plot(tt, dE_derived, "k--", lw=1.5, label="derived formula")
ax.plot(tt, dE_exact, color="tab:red", lw=2, label="exact (closed-form solution)")
ax.axhline(0, color="gray", lw=0.5)
ax.set(xlabel="t [s]", ylabel="ΔE per step [J]",
       title=f"Semi-implicit Euler ΔE (m={m}, k={k}, c={c}, dt={dt})")
ax.legend()
fig.tight_layout()
fig.savefig("semi_implicit_three_lines_dE.png", dpi=140)