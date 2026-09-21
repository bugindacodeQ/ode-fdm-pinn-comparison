# Exact, Finite-Difference, and PINN Solutions of a Boundary-Value ODE

This repository is a reproducible numerical study of a second-order boundary-value problem solved in three complementary ways:

1. an exact analytical solution;
2. a second-order central finite-difference method (FDM); and
3. a physics-informed neural network (PINN).

The exact solution provides a reference against which the numerical methods can be verified. The finite-difference solution serves as a classical deterministic baseline, while the PINN demonstrates a mesh-free optimization-based approach that incorporates the governing equation directly into its training objective.

> **Project status:** the complete exact/FDM/PINN experiment, validation suite, structured result export, plotting pipeline, and command-line runner are implemented.

## Problem statement

Find $y:[0,1]\rightarrow\mathbb{R}$ satisfying

```math
\frac{d^2y}{dx^2}+x^2=0, \qquad 0\leq x\leq 1,
```

subject to the Dirichlet boundary conditions

```math
y(0)=2, \qquad y(1)=4.
```

Although these values are sometimes informally called initial conditions, they are prescribed at two different points of the domain and therefore define a two-point boundary-value problem.

## Analytical reference solution

Rearranging and integrating the differential equation twice gives

```math
y''(x)=-x^2,
```

```math
y'(x)=-\frac{x^3}{3}+C_1,
```

and

```math
y(x)=-\frac{x^4}{12}+C_1x+C_2.
```

Applying $y(0)=2$ yields $C_2=2$. Applying $y(1)=4$ then gives $C_1=25/12$. The exact solution is therefore

```math
\boxed{y_{\mathrm{exact}}(x)=-\frac{x^4}{12}+\frac{25}{12}x+2.}
```

This expression is the ground-truth reference for all error and convergence measurements in the project.

## Numerical methods

### Finite-difference method

Partition $[0,1]$ into $N$ equal subintervals with

```math
h=\Delta x=\frac{1}{N}, \qquad x_i=ih, \qquad i=0,1,\ldots,N.
```

At each interior grid point, approximate the second derivative using the centered stencil

```math
y''(x_i)\approx\frac{y_{i-1}-2y_i+y_{i+1}}{h^2}.
```

Substitution into the governing equation produces

```math
y_{i-1}-2y_i+y_{i+1}=-h^2x_i^2,
\qquad i=1,\ldots,N-1.
```

Equivalently, the positive-definite form is

```math
-y_{i-1}+2y_i-y_{i+1}=h^2x_i^2.
```

Together with $y_0=2$ and $y_N=4$, these equations define a tridiagonal linear system for the $N-1$ unknown interior values. The boundary values are moved into the right-hand side of the first and last interior equations.

The centered approximation is second-order accurate, so the discretization error is expected to satisfy

```math
\lVert y_h-y_{\mathrm{exact}}\rVert=\mathcal{O}(h^2)
```

when the grid is successively refined.

### Physics-informed neural network

Let $N_\theta(x)$ be a neural network with trainable parameters $\theta$. Smooth activation functions are required because the differential equation contains a second derivative with respect to $x$.

Two common treatments of the boundary conditions are possible.

#### Soft constraints

For an unconstrained approximation $y_\theta(x)$, define the differential-equation residual

```math
r_\theta(x)=\frac{d^2y_\theta}{dx^2}+x^2.
```

At collocation points $\{x_f^{(i)}\}_{i=1}^{N_f}\subset(0,1)$, the physics loss is

```math
\mathcal{L}_{\mathrm{physics}}
=\frac{1}{N_f}\sum_{i=1}^{N_f}
\left[r_\theta\left(x_f^{(i)}\right)\right]^2.
```

The boundary penalty is

```math
\mathcal{L}_{\mathrm{boundary}}
=\left[y_\theta(0)-2\right]^2
+\left[y_\theta(1)-4\right]^2,
```

and the total objective is

```math
\mathcal{L}(\theta)
=\mathcal{L}_{\mathrm{physics}}
+\lambda_{\mathrm{BC}}\mathcal{L}_{\mathrm{boundary}}.
```

This formulation is useful as a baseline, but its behavior depends on the boundary-loss weight $\lambda_{\mathrm{BC}}$.

#### Hard constraints

The preferred formulation embeds the boundary conditions directly into the trial solution:

```math
\boxed{
\widehat{y}_\theta(x)=2+2x+x(1-x)N_\theta(x).
}
```

The linear term $2+2x$ interpolates the two boundary values, and the factor $x(1-x)$ vanishes at both endpoints. Consequently,

```math
\widehat{y}_\theta(0)=2,
\qquad
\widehat{y}_\theta(1)=4
```

for every value of $\theta$, including before training.

Differentiating the trial solution gives

```math
\widehat{y}_\theta''(x)
=-2N_\theta(x)
+2(1-2x)N_\theta'(x)
+x(1-x)N_\theta''(x).
```

The residual optimized by the PINN is therefore

```math
r_\theta(x)
=-2N_\theta(x)
+2(1-2x)N_\theta'(x)
+x(1-x)N_\theta''(x)
+x^2,
```

with objective

```math
\mathcal{L}(\theta)
=\frac{1}{N_f}\sum_{i=1}^{N_f}
\left[r_\theta\left(x_f^{(i)}\right)\right]^2.
```

Automatic differentiation should be used to compute the required derivatives. The hard-constrained form is the primary PINN formulation for this project because it eliminates boundary-condition error by construction and removes the need to tune $\lambda_{\mathrm{BC}}$.

## Experimental protocol

All methods should be evaluated on a common, dense set of points in $[0,1]$. Report at least the following quantities:

- maximum absolute error, $L_\infty$;
- root-mean-square error;
- relative $L_2$ error;
- differential-equation residual RMSE; and
- absolute boundary errors at $x=0$ and $x=1$.

For the finite-difference method, run a grid-refinement study and estimate the observed convergence order from consecutive errors. The expected order is approximately two.

For the PINN, record the network architecture, activation function, optimizer, learning-rate schedule, collocation strategy, number of iterations, numerical precision, random seed, and hardware. Because neural-network training is stochastic, results should be summarized over multiple seeds rather than from a single favorable run.

Plots should include:

- exact, finite-difference, and PINN solutions on the same axes;
- pointwise absolute error for each numerical method;
- PINN residual over the domain;
- PINN loss history; and
- finite-difference error against grid spacing on logarithmic axes.

## Repository layout

```text
.
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- exact_solution.py
|   |-- finite_difference.py
|   |-- pinn.py
|   |-- plotting.py
|   |-- experiment.py
|   `-- compare_solutions.py
|-- tests/
|   |-- __init__.py
|   |-- test_exact_solution.py
|   |-- test_finite_difference.py
|   |-- test_pinn.py
|   |-- test_compare_solutions.py
|   |-- test_plotting.py
|   `-- test_experiment.py
`-- outputs/
    |-- data/
    `-- figures/
```

### Module responsibilities

| Path | Responsibility |
|---|---|
| `src/config.py` | Shared domain, discretization, training, precision, and random-seed settings. |
| `src/exact_solution.py` | Analytical reference solution and exact derivatives. |
| `src/finite_difference.py` | Grid construction, tridiagonal system assembly, and numerical solve. |
| `src/pinn.py` | Network definition, constrained trial solution, automatic differentiation, training, and inference. |
| `src/compare_solutions.py` | Common evaluation grid, metrics, convergence studies, and result export. |
| `src/plotting.py` | Headless generation of solution, error, residual, loss, and convergence figures. |
| `src/experiment.py` | Reproducible end-to-end orchestration and command-line configuration. |
| `tests/test_exact_solution.py` | Verification of the differential equation and both boundary values. |
| `tests/test_finite_difference.py` | Matrix assembly, boundary handling, numerical accuracy, and convergence tests. |
| `tests/test_pinn.py` | Trial-function constraints, derivative flow, residual shape, and training smoke tests. |
| `outputs/data/` | Machine-readable numerical results and experiment metadata. |
| `outputs/figures/` | Publication-ready comparison, error, residual, and convergence figures. |

## Running the experiment

Install the declared dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

For an editable installation with test dependencies, use:

```powershell
python -m pip install -e ".[test]"
```

Run the complete experiment with the documented defaults:

```powershell
python -m src.compare_solutions
```

The command trains the PINN, evaluates all methods on a common grid, performs the finite-difference refinement study, and writes all CSV, JSON, and PNG artifacts under `outputs/`.

Key settings can be overridden explicitly:

```powershell
python -m src.compare_solutions --epochs 2000 --hidden-layers 16,16 --seed 42
```

View every available option with:

```powershell
python -m src.compare_solutions --help
```

Run the complete validation suite with:

```powershell
python -m pytest tests -v
```

## Reproducibility requirements

The implementation should satisfy the following engineering requirements:

- use one shared configuration for the domain and boundary values;
- set and record all random seeds used by the PINN;
- use a consistent floating-point precision across training and evaluation;
- keep generated artifacts out of the source directories;
- export numerical results in a machine-readable format;
- include environment and dependency versions with reported experiments; and
- avoid selecting hyperparameters using the exact solution, except in explicitly labeled diagnostic studies.

## Verification criteria

The study is considered complete when:

1. the analytical expression satisfies the ODE and both boundary conditions to numerical precision;
2. the finite-difference linear system enforces the boundary values correctly;
3. finite-difference errors decrease at approximately second order under grid refinement;
4. the hard-constrained PINN satisfies both boundary values to numerical precision for arbitrary network parameters;
5. the trained PINN achieves a low residual throughout the domain, not only at its training points; and
6. all reported errors are computed against the same exact reference on the same evaluation grid.

## Scope

This problem is intentionally small enough to admit a closed-form solution. Its purpose is not to show that a PINN is more efficient than a classical solver for this ODE. Instead, it provides a controlled benchmark for studying discretization error, residual minimization, boundary-condition enforcement, optimization variability, and reproducible comparison between conventional numerical methods and physics-informed learning.
