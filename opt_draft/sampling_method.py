# Objective: min entropy of density matrix S in C^2
# Constraints: tr(S)=1, S>=0

# implementation of random sampling method

import numpy as np
import math

# 1. generate random density matrix
# rho = S S^* / tr(S S^*)
def generate_matrix():
    S = np.random.randn(2,2) + 1j*np.random.randn(2,2)
    SS = S @ S.conj().T
    rho = SS / np.trace(SS)
    return rho
 
# 2. find entropy
# H = -sum(lambda_i * log(lambda_i))
def find_entropy(S):
    eigvals = np.linalg.eigvalsh(S)
    entropy = -np.sum(eigvals * np.log(eigvals + 1e-12)) # avoid log(0)
    return entropy

# main
N = 1000
min_entropy = math.inf
best_matrix = None

for _ in range(N):
    rho = generate_matrix()
    entropy = find_entropy(rho)
    if entropy < min_entropy:
        min_entropy = entropy
        best_matrix = rho

print("Minimum Entropy:", min_entropy)
print("Best Density Matrix:\n", best_matrix)
print("Eigenvalues of Best Density Matrix:", np.linalg.eigvalsh(best_matrix))