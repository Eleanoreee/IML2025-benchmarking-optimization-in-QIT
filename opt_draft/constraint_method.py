# Objective: min entropy of density matrix S in C^2
# depolarizing channel: (1-p)*rho + p*I/2
# we use constrained optimization method by computing eigenvalues

import numpy as np
from scipy.optimize import minimize

# ---- parameters ----
p = 0.10   # depolarizing prob

# rho(a,x,y)
def rho_of(a, x, y):
    return np.array([[a,       x + 1j*y],
                     [x - 1j*y, 1.0 - a]], dtype=complex)

# von Neumann entropy in bits for depolarized state
def entropy_dep(a, x, y):
    rho = rho_of(a, x, y)
    # eigenvalues of depolarized state
    lam = np.linalg.eigvalsh(rho)                 # λ_i
    mu  = (1 - p) * lam + 0.5 * p                 # μ_i
    mu  = np.clip(mu.real, 1e-15, 1.0)            # numeric safety
    return float(-np.sum(mu * np.log2(mu)))

# objective wrapper for minimize
def f_obj(vars):
    a, x, y = vars
    return entropy_dep(a, x, y)

# ---- constraints ----
# 1) bounds for a: 0 <= a <= 1
bounds = [(0.0, 1.0),   # a
          (None, None), # x
          (None, None)] # y

# 2) PSD inequality: a(1-a) - (x^2 + y^2) >= 0
def g_psd(vars):
    a, x, y = vars
    return a*(1.0 - a) - (x*x + y*y)

ineq_cons = {'type': 'ineq', 'fun': g_psd}

# ---- starting point ----
# take a mixed feasible point: a=0.5, x=y=0 (maximally mixed)
x0 = np.array([0.5, 0.0, 0.0])

# ---- solve ----
# SLSQP works well for smooth inequality constraints.
res = minimize(f_obj, x0, method='SLSQP', bounds=bounds, constraints=[ineq_cons],
               options={'ftol':1e-12, 'maxiter':1000})

a_opt, x_opt, y_opt = res.x
rho_opt = rho_of(a_opt, x_opt, y_opt)
evals_opt = np.linalg.eigvalsh(rho_opt)

print("status:", res.message)
print("min entropy (bits):", res.fun)
print("a, x, y:", a_opt, x_opt, y_opt)
print("eigs(ρ):", np.sort(evals_opt)[::-1])
print("PSD slack a(1-a)-x^2-y^2:", g_psd(res.x))
