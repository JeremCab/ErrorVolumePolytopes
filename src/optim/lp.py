import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import OptimizeResult

def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, 
            method=None, x0=None, options=None, **kwargs):
    """
    Gurobi drop-in replacement for scipy.optimize.linprog.
    """
    c = np.array(c, dtype=float)
    num_vars = len(c)

    # 1. Parse bounds (SciPy default is (0, None) for all variables)
    lb = np.zeros(num_vars)
    ub = np.full(num_vars, GRB.INFINITY)
    
    if bounds is not None:
        # Check if bounds is a single tuple applied to all variables
        if len(bounds) == 2 and not isinstance(bounds[0], (tuple, list, np.ndarray)):
            bounds = [bounds] * num_vars
            
        for i, (l, u) in enumerate(bounds):
            lb[i] = -GRB.INFINITY if l is None else l
            ub[i] = GRB.INFINITY if u is None else u

    # 2. Initialize Gurobi model (suppressing console output to match SciPy default)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    m = gp.Model("linprog", env=env)

    # 3. Add variables using Gurobi's matrix API
    x = m.addMVar(shape=num_vars, lb=lb, ub=ub)

    # 4. Set objective
    m.setObjective(c @ x, GRB.MINIMIZE)

    # 5. Add inequality constraints (A_ub * x <= b_ub)
    if A_ub is not None and b_ub is not None:
        A_ub = np.array(A_ub, dtype=float)
        b_ub = np.array(b_ub, dtype=float)
        if A_ub.size > 0:
            m.addConstr(A_ub @ x <= b_ub, name="ineq")

    # 6. Add equality constraints (A_eq * x == b_eq)
    if A_eq is not None and b_eq is not None:
        A_eq = np.array(A_eq, dtype=float)
        b_eq = np.array(b_eq, dtype=float)
        if A_eq.size > 0:
            m.addConstr(A_eq @ x == b_eq, name="eq")

    # 7. Optimize
    m.optimize()

    # 8. Format the output to match SciPy's OptimizeResult
    res = OptimizeResult()
    if m.status == GRB.OPTIMAL:
        res.success = True
        res.status = 0
        res.message = "Optimization terminated successfully."
        res.x = x.X
        res.fun = m.ObjVal
    elif m.status == GRB.INFEASIBLE:
        res.success = False
        res.status = 2
        res.message = "The problem is infeasible."
        res.x = None
        res.fun = None
    elif m.status == GRB.UNBOUNDED:
        res.success = False
        res.status = 3
        res.message = "The problem is unbounded."
        res.x = None
        res.fun = None
    else:
        res.success = False
        res.status = 4
        res.message = f"Optimization failed with Gurobi status code {m.status}."
        res.x = None
        res.fun = None

    return res