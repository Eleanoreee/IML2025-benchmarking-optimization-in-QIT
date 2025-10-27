import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----- global parameters -----
n = 3  # dimension of matrix
p = 0.10    # parameter of depolarization channel

I = np.eye(n)
solvers = ["BFGS", "Nelder-Mead", "Powell"] # some solvers
num_problems = 10
tau = 1e-7  # tolerance
budget = 500   # max function evals per run


# ----- entropy and depolarizing -----
def entropy(rho):
    w = np.linalg.eigvalsh(rho)
    w = np.clip(w.real, 1e-12, 1)
    return -float(np.sum(w * np.log2(w)))

def depolarize(rho):
    return (1 - p) * rho + (p / n) * I


# ----- objective wrapper (to record trajectory) -----
def make_objective_tracker():
    evals, values = [], []  # number of evals, corresponding entropy
    def objective(params):  # obj func
        real_part = params[:n*n].reshape(n, n)
        imag_part = params[n*n:].reshape(n, n)
        Y = real_part + 1j * imag_part
        rho = Y.conj().T @ Y
        rho /= np.trace(rho)
        val = entropy(depolarize(rho))
        evals.append(len(evals) + 1)
        values.append(val)
        return val
    return objective, evals, values

records = []


# ----- run solvers -----
# for each problems
for pid in range(num_problems):
    # random initial matrix
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    init = np.concatenate([X.real.flatten(), X.imag.flatten()])

    f0 = None   # init
    f_best = np.inf
    run_data = {}

    # run all solvers
    for solver in solvers:
        obj, evals, values = make_objective_tracker()
        res = minimize(obj, init, method=solver,
                       options={'maxiter': budget, 'disp': False})
        if f0 is None:
            f0 = values[0]
        f_best = min(f_best, np.min(values))
        run_data[solver] = (evals, values)

    # compute f_L and target
    f_L = f_best    # local min obtained by all solvers
    f_target = f_L + tau * (f0 - f_L)   # apply tol

    # find t_{p,s} for each solver: how many func evals needed for solver s to reach convergence
    for solver in solvers:
        evals, values = run_data[solver]
        tps = np.inf
        for e, v in zip(evals, values):
            if v <= f_target:
                tps = e
                break
        records.append({    # record as df
            "problem": pid,
            "solver": solver,
            "tps": tps,
            "fL": f_L,
            "ftarget": f_target,
            "f0": f0
        })

df = pd.DataFrame(records)
print(df)


# ----- performance profile -----
def performance_profile(df, save_dir="plots", alpha_max=10):
    problems = df["problem"].unique()
    solvers = df["solver"].unique()
    alpha_vals = np.linspace(1, alpha_max, 200)
    perf = {s: [] for s in solvers}

    for alpha in alpha_vals:
        for s in solvers:
            count = 0
            for p in problems:
                sub = df[df.problem == p]
                t_best = sub["tps"].min()
                t_s = sub[sub.solver == s]["tps"].values[0]
                if np.isfinite(t_s) and t_s <= alpha * t_best:
                    count += 1
            perf[s].append(count / len(problems))

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6,4))
    for s in solvers:
        plt.plot(alpha_vals, perf[s], label=s)
    plt.xlabel("α (relative factor to best t_ps)")
    plt.ylabel("ρ_s(α)")
    plt.title("Performance Profile (Moré–Wild Definition)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(save_dir, "performance_profile_MW.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Saved to {path}")



# ----- data profile -----
def data_profile(df, n_vars, save_dir="plots", K_max=500):
    problems = df["problem"].unique()
    solvers = df["solver"].unique()
    K_vals = np.linspace(0, K_max, 200)
    data = {s: [] for s in solvers}

    for K in K_vals:
        for s in solvers:
            count = 0
            for p in problems:
                t_s = df[(df.problem == p) & (df.solver == s)]["tps"].values[0]
                if np.isfinite(t_s) and t_s <= K * (n_vars + 1):
                    count += 1
            data[s].append(count / len(problems))

    plt.figure(figsize=(6, 4))
    for s in solvers:
        plt.plot(K_vals, data[s], label=s)
    plt.xlabel("K (budget per dimension)")
    plt.ylabel("d_s(K)")
    plt.title("Data Profile")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, "data_profile_MW.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Data profile saved to {save_path}")


# ----- run and save -----
n_vars = 2 * n * n  # number of parameters
performance_profile(df, save_dir="plots")
data_profile(df, n_vars=n_vars, save_dir="plots")