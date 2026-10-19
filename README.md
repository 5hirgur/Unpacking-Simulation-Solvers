# Unpacking Simulation Solvers

Code and data for the ROSCon 2026 talk **Unpacking Simulation Solvers** (22 Oct 2026).

"Which physics engine?" is often the wrong question: the solver underneath decides whether simulated objects stand, slide or explode. This repo holds the experiments behind the talk, built on [Newton](https://github.com/newton-physics/newton).

## Experiments

### 1. Newton vs conjugate gradient (MuJoCo solver)
A stack of N cubes (N = 3, 5, 7, 10, 15) is simulated for 3 s at dt = 5 ms (600 steps). The variable is the **iteration cap**, the maximum solver iterations per time step. A stack "stands" if the final sag and the maximum slide are both under 20 mm.

| N cubes | Lowest cap where Newton stands | Lowest cap where CG stands |
|---|---|---|
| 3 | 1–2 | 3 |
| 5 | 1–2 | 5 |
| 7 | 1–2 | 5–8 |
| 10 | 1–2 | 8 |
| 15 | 1–2 | 12–50 (non-monotonic) |

Both solvers converge to the same sag when they stand. Near the threshold, CG results varied from run to run on GPU.

- Script: `TODO: path/newton_stack.py` (usage: `TODO`)
- Data: `TODO: path to *_sweep.csv, *_timeseries.csv`
- Figures: `TODO: fig1–fig4`

### 2. XPBD: iterations change stiffness
One 1 kg particle hanging from a spring (k = 100 N/m, dt = 1/60 s, g = 10). The physical stretch is mg/k = 0.1 m.

| `iterations` | 1 | 2 | 5 | 10 | 20 | 50 | 200 |
|---|---|---|---|---|---|---|---|
| stretch (m) | 0.1 | 0.0493 | 0.0189 | 0.0088 | 0.0038 | 0.00095 | 0.00001 |

In Newton's `SolverXPBD`, the effective stiffness grows with `iterations`. Closed form: s_n = g·dt²·rⁿ/(1−rⁿ), r = α̃/(1+α̃), α̃ = 1/(k·dt²).
Scope: single particle, undamped spring, CPU. Tetrahedra, joints and cloth were not run.

- Script: `TODO: path/xpbd_spring.py`

### 3. SemiImplicit: stability
The same spring with `SolverSemiImplicit`. It reproduces mg/k when ω·dt is small (ω = √(k/m)) and diverges when ω·dt is large (observed unstable at ω·dt = 1.67 with damping, and at 2.5).

- Script: `TODO: path/semi_implicit_spring.py`

## Solver summary

| Solver | Coordinates | Core idea | Key setting |
|---|---|---|---|
| SemiImplicit | Maximal | Evaluate forces, then symplectic Euler | `dt`; model `ke`/`kd` |
| Featherstone | Reduced | Assemble the mass matrix, solve with Cholesky | none (non-iterative) |
| XPBD | Maximal |
