"""MILP solver dispatch for BOPOs ILP.

Two backends:
  - "glpk":  cvxopt.glpk.ilp (legacy, paper baseline)
  - "highs": HiGHS via highspy (5-10x faster on Mittelmann MILP benchmark)

Both solve the same standard-form MILP and return the same optimal x when the
problem has a unique optimum. Selected via env var ``PREORDER_SOLVER`` (or
:func:`set_solver`); default is "glpk" so the paper baseline is unchanged.
"""

from __future__ import annotations

import os
import numpy as np


_SOLVER_ENV = "PREORDER_SOLVER"


def get_solver() -> str:
    return os.environ.get(_SOLVER_ENV, "glpk").lower()


def set_solver(name: str) -> None:
    if name.lower() not in ("glpk", "highs"):
        raise ValueError(f"Unknown solver: {name}; expected glpk or highs")
    os.environ[_SOLVER_ENV] = name.lower()


def solve_milp(c, G, h, A, b, I, B):
    """Solve MILP: min c^T x  s.t.  Gx <= h,  Ax = b,  x_i ∈ {0,1} for i ∈ B,
    x_i ∈ ℤ for i ∈ I.

    Inputs are numpy arrays (c is (n,1); G is (m,n); h is (m,1); A is (k,n);
    b is (k,1)) and I, B are sets of column indices.

    Returns (status, x) where x is an (n,1) numpy column vector matching the
    cvxopt convention used at call sites.
    """
    solver = get_solver()
    if solver == "glpk":
        return _solve_glpk(c, G, h, A, b, I, B)
    return _solve_highs(c, G, h, A, b, I, B)


def _solve_glpk(c, G, h, A, b, I, B):
    from cvxopt import matrix
    from cvxopt.glpk import ilp

    status, x = ilp(matrix(c), matrix(G), matrix(h), matrix(A), matrix(b), I, B)
    return status, np.array(x)


def _solve_highs(c, G, h, A, b, I, B):
    """HiGHS via scipy.optimize.milp (officially wraps HiGHS, stable API)."""
    from scipy.optimize import LinearConstraint, milp, Bounds

    # cvxopt.glpk silently uses only the first ncol(G/A) entries of c, so the
    # legacy encoders allocate c larger than the constraint matrix expects.
    # Mirror that behavior: trim c to the constraint width before solving.
    G_cols = np.asarray(G).shape[1] if np.asarray(G).ndim >= 2 and np.asarray(G).size else 0
    A_cols = np.asarray(A).shape[1] if np.asarray(A).ndim >= 2 and np.asarray(A).size else 0
    constraint_cols = max(G_cols, A_cols)
    n = constraint_cols if constraint_cols else int(np.asarray(c).shape[0])
    c_arr = np.asarray(c, dtype=np.float64).reshape(-1)[:n]

    # Variable bounds:
    #   binary i ∈ B  -> [0, 1] integer
    #   integer i ∈ I (not B) -> [0, +inf) integer
    #   continuous (rest) -> match cvxopt.glpk default, which treats unspecified
    #     vars as unbounded reals. The BOPOs ILPs here have all variables in B,
    #     so this branch never triggers in practice, but keep correctness.
    lb = np.full(n, -np.inf, dtype=np.float64)
    ub = np.full(n, np.inf, dtype=np.float64)
    integrality = np.zeros(n, dtype=np.int32)  # 0=continuous in scipy.milp
    for i in B:
        lb[i] = 0.0
        ub[i] = 1.0
        integrality[i] = 1
    for i in I:
        if i not in B:
            lb[i] = 0.0
            integrality[i] = 1

    G_arr = np.asarray(G, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64).reshape(-1)
    A_arr = np.asarray(A, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64).reshape(-1)

    # scipy.milp requires 2D A and full-length lb/ub arrays per constraint.
    if G_arr.ndim == 1:
        G_arr = G_arr.reshape(1, -1)
    if A_arr.ndim == 1:
        A_arr = A_arr.reshape(1, -1)
    constraints = []
    if G_arr.size:
        constraints.append(
            LinearConstraint(G_arr, np.full(G_arr.shape[0], -np.inf), h_arr)
        )
    if A_arr.size:
        constraints.append(LinearConstraint(A_arr, b_arr, b_arr))

    _time_limit = os.environ.get("HIGHS_TIME_LIMIT")
    _mip_gap = os.environ.get("HIGHS_MIP_REL_GAP")
    _options = {}
    if _time_limit is not None:
        _options["time_limit"] = float(_time_limit)
    if _mip_gap is not None:
        _options["mip_rel_gap"] = float(_mip_gap)

    res = milp(
        c=c_arr,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lb=lb, ub=ub),
        options=_options if _options else None,
    )
    if res.x is None:
        # HiGHS hit the time limit before finding any integer-feasible
        # solution (common on K>=50 BOPOs ILPs at HIGHS_TIME_LIMIT<=5s).
        # Fall back to GLPK which has no built-in time limit. If GLPK
        # also struggles, _solve_glpk would block — accept that worst
        # case to avoid losing the entire (instance, IA) result.
        return _solve_glpk(c, G, h, A, b, I, B)
    x = np.asarray(res.x, dtype=np.float64).reshape(-1, 1)
    return res.status, x
