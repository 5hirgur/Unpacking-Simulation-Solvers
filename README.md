# Unpacking Simulation Solvers

Code for the ROSCon 2026 talk **Unpacking Simulation Solvers** (22 Oct 2026).

"Which physics engine?" is often the wrong question: the solver underneath decides whether simulated objects stand, slide or explode. This repo contains the experiment behind the solver comparison in the talk, built on [Newton](https://github.com/newton-physics/newton) and its MuJoCo solver.

## Experiment: Newton vs conjugate gradient

The MuJoCo solver offers several ways to solve contact. This experiment compares two of them:

- **Newton:** takes a full Newton step with a Cholesky solve each iteration.
- **Conjugate Gradient (CG):** uses only the gradient and takes smaller steps.

**Setup.** A stack of N cubes is simulated for 3 s at a 5 ms time step. The variable is the **iteration cap**, the maximum number of solver iterations allowed per time step. For each cap we measure:

- **sag:** how far the stack has compressed at the end of the run.
- **slide:** the largest sideways movement of the cubes during the run.
- **stands:** the stack counts as standing if sag and slide are both under 20 mm.

The experiment asks how many iterations each solver needs before a stack of a given height stays standing.
