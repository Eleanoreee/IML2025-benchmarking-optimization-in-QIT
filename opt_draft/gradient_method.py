# Objective: min entropy of density matrix S in C^2
# depolarizing channel: (1-p)*rho + p*I/2
# Constraints: tr(S)=1, S>=0

# implementation of unconstrained optimization method: gradient-based

import numpy as np
from scipy.optimize import minimize

p = 0.10
I = np.eye(2)

# 1. entropy
def entropy(r):
    w = np.linalg.eigvalsh(r) # find complex eigenvalues
    w = np.clip(w, 1e-12, 1) # aviod log(0)
    return -float(np.sum(w * np.log2(w)))

# 2. depolarizing channel
def dep(r):
    return (1 - p) * r + 0.5 * p * I

# 3. obj function (transpose to get hermitian)
def objective_function(params):
    Y = (params[:4] + 1j * params[4:]).reshape(2, 2) # 2x2 complex matrix
    rho = Y.conj().T @ Y
    rho /= np.trace(rho)
    val = entropy(dep(rho))
    return val

# optimization loop
X = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
initial_guess = np.concatenate([X.real.flatten(), X.imag.flatten()])
result = minimize(objective_function, initial_guess, method='BFGS') # 'BFGS' is a kind of newton's method works for smooth functions

# extract optimal density matrix
opt_params = result.x
Y_best = (opt_params[:4] + 1j * opt_params[4:]).reshape(2, 2)
rho_best = Y_best.conj().T @ Y_best
rho_best /= np.trace(rho_best)

print("Minimized output entropy S(dep(rho)) =", result.fun)
print("Eigenvalues of optimal rho:", np.linalg.eigvalsh(rho_best))
print("Optimal rho:\n", rho_best)