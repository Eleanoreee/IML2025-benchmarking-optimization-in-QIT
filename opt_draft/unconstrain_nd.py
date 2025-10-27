import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----- global parameters -----
n = 3
p = 0.10
I = np.eye(n)
solvers = ["BFGS", "Nelder-Mead", "Powell"]
num_problems = 10

# ----- entropy and depolarizing -----
def entropy(rho):
    w = np.linalg.eigvalsh(rho)
    w = np.clip(w.real, 1e-12, 1)
    return -float(np.sum(w * np.log2(w)))

def depolarize(rho):
    return (1 - p) * rho + (p / n) * I

# ----- objective with function-eval counter -----
f_eval_counter = [0]

def objective(params):
    f_eval_counter[0] += 1
    real_part = params[:n*n].reshape(n, n)
    imag_part = params[n*n:].reshape(n, n)
    Y = real_part + 1j * imag_part
    rho = Y.conj().T @ Y
    rho /= np.trace(rho)
    return entropy(depolarize(rho))

# ----- storage -----
records = []

# loop over problems
for pid in range(num_problems):
    # random initial matrix
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    init = np.concatenate([X.real.flatten(), X.imag.flatten()])

    for solver in solvers:
        f_eval_counter[0] = 0
        res = minimize(objective, init, method=solver,
                       options={'maxiter': 500, 'disp': False})
        record = {
            "problem": pid,
            "solver": solver,
            "fun": res.fun, # minimum entropy found
            "nfev": res.nfev if hasattr(res, "nfev") else f_eval_counter[0], # function evaluations
            "success": res.success, # whether converged
        }
        records.append(record)

df = pd.DataFrame(records)
print(df)

# ----- performance_profile -----
def performance_profile(df, save_dir="plots", alpha_max=10.0):
    problems = df["problem"].unique()
    solvers = df["solver"].unique()
    alpha_vals = np.linspace(1, alpha_max, 200)
    perf = {s: [] for s in solvers}

    for alpha in alpha_vals:
        for s in solvers:
            count = 0
            for p in problems:
                sub = df[df.problem == p]
                best_t = sub["nfev"].min()
                t_s = sub[sub.solver == s]["nfev"].values[0]
                if t_s <= alpha * best_t:
                    count += 1
            perf[s].append(count / len(problems))

    # Plot
    plt.figure(figsize=(6,4))
    for s in solvers:
        plt.plot(alpha_vals, perf[s], label=s)
    plt.xlabel("α (relative factor to best time)")
    plt.ylabel("ρ_s(α)")
    plt.title("Performance Profile")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, "performance_profile.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ----- data_profile -----
def data_profile(df, n_vars, save_dir="plots", K_max=500):
    problems = df["problem"].unique()
    solvers = df["solver"].unique()
    K_vals = np.linspace(0, K_max, 200)
    data = {s: [] for s in solvers}

    for K in K_vals:
        for s in solvers:
            count = 0
            for p in problems:
                t_s = df[(df.problem==p)&(df.solver==s)]["nfev"].values[0]
                if t_s <= K * (n_vars + 1):
                    count += 1
            data[s].append(count / len(problems))

    # Plot
    plt.figure(figsize=(6,4))
    for s in solvers:
        plt.plot(K_vals, data[s], label=s)
    plt.xlabel("K (budget per dimension)")
    plt.ylabel("d_s(K)")
    plt.title("Data Profile")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, "data_profile.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


n_vars = 2 * n * n
performance_profile(df, save_dir="/Users/eleanorewu/Desktop/IML/min_entropy", alpha_max=10)
data_profile(df, n_vars=n_vars, save_dir="/Users/eleanorewu/Desktop/IML/min_entropy", K_max=500)