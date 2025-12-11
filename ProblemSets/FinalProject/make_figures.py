# make_figures.py
# Generate figures/graphs using results from solve_model.py


import os
import numpy as np
import matplotlib.pyplot as plt

# Import the script which computes the model; this will run the VFI
import solve_model

# Create output directory
out_dir = os.path.abspath('.')

# 1) Wage PDF
try:
    wages = solve_model.wages
    p_w = solve_model.p_w
    plt.figure()
    plt.scatter(wages, p_w, s=6)
    plt.title('Wage PDF')
    plt.xlabel('Wage')
    plt.ylabel('Probability')
    plt.savefig(os.path.join(out_dir, 'wage_pdf.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved wage_pdf.png')
except AttributeError:
    print('wages or p_w not found in solve_model module')

# 2) Eps PDF
try:
    eps_grid = solve_model.eps_grid
    eps_p = solve_model.eps_p
    plt.figure()
    plt.scatter(eps_grid, eps_p, s=6)
    plt.title('Financial Income (eps) PDF')
    plt.xlabel('eps')
    plt.ylabel('Probability')
    plt.savefig(os.path.join(out_dir, 'eps_pdf.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved eps_pdf.png')
except AttributeError:
    print('eps_grid or eps_p not found in solve_model module')

# 3) Convergence diagnostics (plots of V_matrix)
try:
    V_matrix = solve_model.V_matrix
    wages = solve_model.wages
    plt.figure()
    ncols = V_matrix.shape[1]
    # plot up to first 5 columns (initial + 4 iterations) if available
    cols_to_plot = min(5, ncols)
    labels = ['Initial guess'] + [f'{i}th iter' for i in range(1, cols_to_plot)]
    for i in range(cols_to_plot):
        plt.plot(wages, V_matrix[:, i], label=labels[i])
    plt.plot(wages, solve_model.V, label='Final V', lw=2, color='k')
    plt.legend()
    plt.title('Value function convergence')
    plt.xlabel('Wage')
    plt.ylabel('V(w)')
    plt.savefig(os.path.join(out_dir, 'convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved convergence.png')
except AttributeError:
    print('V_matrix or V not found in solve_model module')

# 4) Value function and policy
try:
    wages = solve_model.wages
    V = solve_model.V
    phi_policy = solve_model.phi_policy
    w_reserve = getattr(solve_model, 'w_reserve', np.nan)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(wages, V, lw=1.5)
    plt.xlabel('wage w')
    plt.ylabel('V(w)')
    plt.title('Value function V(w)')

    plt.subplot(1,2,2)
    plt.plot(wages, phi_policy.astype(int), drawstyle='steps-post')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('wage w')
    plt.ylabel('policy (1=accept)')
    plt.title('Acceptance policy (1 = accept)')
    if not np.isnan(w_reserve):
        plt.axvline(w_reserve, color='red', linestyle='--', label=f'res.wage={w_reserve:.2f}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'value_policy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved value_policy.png')
except AttributeError:
    print('V, phi_policy or wages not found in solve_model module')

# 5) Analytic reservation wage as function of eps (closed-form algebra)
try:
    beta = solve_model.beta
    b = solve_model.b
    phi = solve_model.phi
    V = solve_model.V
    eps_grid = solve_model.eps_grid
    mu_eps = solve_model.mu_eps

    # Compute EV (expected continuation value) as in VFI
    EV = np.sum(V * solve_model.p_w)
    # analytic reservation wage (solving v_accept = v_reject for w):
    # (w + phi*mu_eps)/(1-beta) = b + eps + beta*EV
    # => w = (1-beta)*(b + eps + beta*EV) - phi*mu_eps
    w_star_analytic = (1.0 - beta) * (b + eps_grid + beta * EV) - phi * mu_eps

    plt.figure(figsize=(7,5))
    plt.plot(eps_grid, w_star_analytic, label=r'$w^*(\varepsilon)$', lw=2)
    plt.xlabel(r'Financial income $\varepsilon$')
    plt.ylabel(r'Reservation wage $w^*(\varepsilon)$')
    plt.title('Analytic reservation wage')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'w_star_analytic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved w_star_analytic.png')
except AttributeError:
    print('One of beta, b, phi, V, eps_grid, or mu_eps not found in solve_model module')

# 6) Numerical reservation wage as function of eps (grid search)
try:
    wages = solve_model.wages
    V = solve_model.V
    p_w = solve_model.p_w
    eps_grid = solve_model.eps_grid
    eps_p = solve_model.eps_p
    beta = solve_model.beta
    b = solve_model.b

    W, EPS = np.meshgrid(wages, eps_grid, indexing='ij')
    joint_probs = p_w[:, None] * eps_p[None, :]
    EV = np.sum(V * p_w)
    v_accept = W / (1.0 - beta)
    v_reject = b + EPS + beta * EV

    Neps = eps_grid.size
    w_star = np.empty(Neps)
    for j in range(Neps):
        accept_better = v_accept[:, j] >= v_reject[:, j]
        if accept_better.any():
            w_star[j] = wages[np.where(accept_better)[0][0]]
        else:
            w_star[j] = wages[-1]

    plt.figure(figsize=(7,5))
    plt.plot(eps_grid, w_star)
    plt.xlabel(r'Financial income $\varepsilon$')
    plt.ylabel(r'Reservation wage $w^*(\varepsilon)$')
    plt.title('Numerical reservation wage as function of financial income')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'w_star_numeric.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved w_star_numeric.png')
except Exception as e:
    print('Failed to compute numerical w_star:', str(e))
