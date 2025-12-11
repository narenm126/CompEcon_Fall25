# solve_model.py
# Combined code extracted from FinalProject.ipynb (cells: imports, wage grid, eps grid, VFI)

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time


# ----------------------------
# Wage grid
# ----------------------------
# Create a grid of wages over which to evaluate v(w)
w_min, w_max = 10, 100  # min and max values
Nw = 1000 # number of grid points for wages
wages = np.linspace(w_min, w_max, Nw)  # use linearly spaced grid over the given support


# Define a vector of probabilities
# Each value is the probability of drawing the corresponding wage in
# the wages vector
# Assumption is that wages are drawn from a log normal
# distribution with mean mu and standard deviation sigma
mu = np.log(40)
sigma = 0.5
p_dens = stats.lognorm.pdf(wages, s=sigma, scale=np.exp(mu))
p_w = p_dens / p_dens.sum() # normalize bc wage vector is bounded


# ----------------------------
# Financial income grid and probabilities (G(eps))
# ----------------------------
eps_mu = 0.0
eps_sigma = 10.0

Neps = 1000
eps_min = eps_mu - 3 * eps_sigma
eps_max = eps_mu + 3 * eps_sigma
eps_grid = np.linspace(eps_min, eps_max, Neps)

# approximate probabilities using normal pdf
from scipy.stats import norm
dens = norm.pdf(eps_grid, loc=eps_mu, scale=eps_sigma)
eps_p = dens / dens.sum()  # normalize


mu_eps = np.sum(eps_grid * eps_p)


# ----------------------------
# Initialize VFI variables
# ----------------------------
V = np.zeros(wages.size)            # initial guess for V(w)
V_matrix = V.reshape(Nw, 1)         # track V across iterations
phi_policy = np.zeros(wages.size, dtype=bool)  # policy: True = accept
v_tol = 1e-8
v_dist = 10.0
max_iter = 2000
iter = 0

# Timing
time_start = time.time()

# Model parameters (these are needed in the VFI)
beta = 0.99  # discount factor
phi = 0.5
expected_wage = np.sum(wages * p_w)
theta = 0.2
b = theta * expected_wage  # unemployment benefits

# ----------------------------
# Value function iteration
# Bellman Equation:
# TV(w) = max{ V_accept(w), b + mu_eps + beta * EV }, where
# V_accept(w) = (w + phi * mu_eps) / (1 - beta)
# EV = E_w[V(w)] (under p)
# ----------------------------
while (v_dist > v_tol) & (iter < max_iter):
    # Expected continuation value (E_w[V(w)])
    EV = np.sum(V * p_w)

    # Value of rejecting now: expected current flow b + mu_eps plus continuation beta*EV
    v_reject = b + mu_eps + beta * EV

    # Value if accept at wage w (closed form, using mean of eps)
    v_accept = (wages + phi * mu_eps) / (1.0 - beta)   # vector of length Nw

    # Applying Bellman operator
    TV = np.maximum(v_accept, v_reject)

    # policy: True if accept is strictly better than reject (ties => reject)
    phi_policy = v_accept > v_reject

    # convergence metric
    v_dist = np.abs(V - TV).max()
    if (iter % 10 == 0) or (v_dist < 1e-6):
        print(f"Iteration {iter:4d}, v_dist = {v_dist:.6e}")

    V = TV.copy()
    V_matrix = np.hstack((V_matrix, V.reshape(Nw, 1)))
    iter += 1

time_end = time.time()
print(f"Value function iteration took {time_end - time_start:.3f} seconds and {iter} iterations")
print(f"Final sup norm distance = {v_dist:.6e}")

# ----------------------------
# Diagnostics
# ----------------------------
U_implied = np.sum(V * p_w)
print(f"Implied unemployed value U = E_w[V(w)] = {U_implied:.6f}")

# Reservation wage: first wage where accept is optimal
accept_idx = np.where(phi_policy)[0]
if accept_idx.size > 0:
    w_reserve = wages[accept_idx[0]]
    print(f"Reservation wage (grid) = {w_reserve:.4f}")
else:
    w_reserve = np.nan
    print("No acceptance on grid (phi all False)")

# Acceptance probability under F
accept_prob = np.sum(p_w * phi_policy)
print(f"Acceptance probability under F: {accept_prob:.4f}")

