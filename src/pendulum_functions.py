##############################################################
## Section 0: The required packages
##############################################################

# for computation
import numpy as np
import math
from scipy.stats import linregress

# for code parallelisation
from numba import njit, prange
from joblib import Parallel, delayed

# for plotting figures and visualising the progress bar
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation
import matplotlib
from tqdm import tqdm

##############################################################
## Section 1.1: Solving the damped driven pendulum for a given
## forcing f (autonomous system)
##############################################################

@njit
def rk4_step(theta, v, t, dt, nu, omega, f):
    k1_theta = v
    k1_v = f*math.cos(t) - nu*v - math.sin(theta)

    theta2 = theta + 0.5*dt*k1_theta
    v2 = v + 0.5*dt*k1_v
    k2_theta = v2
    k2_v = f*math.cos(t + 0.5*dt) - nu*v2 - math.sin(theta2)

    theta3 = theta + 0.5*dt*k2_theta
    v3 = v + 0.5*dt*k2_v
    k3_theta = v3
    k3_v = f*math.cos(t + 0.5*dt) - nu*v3 - math.sin(theta3)

    theta4 = theta + dt*k3_theta
    v4 = v + dt*k3_v
    k4_theta = v4
    k4_v = f*math.cos(t + dt) - nu*v4 - math.sin(theta4)

    theta_new = theta + (dt/6)*(k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
    v_new = v + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

    return theta_new, v_new

@njit
def solve_pendulum_numba(theta0, v0, dt, n_steps, nu, omega, f, t0):
    theta = theta0
    v = v0
    t = t0

    out_theta = np.empty(n_steps)
    out_v = np.empty(n_steps)

    for i in range(n_steps):
        out_theta[i] = theta
        out_v[i] = v
        theta, v = rk4_step(theta, v, t, dt, nu, omega, f)
        t += dt

    return out_theta, out_v

def solve_pendulum(theta0, v0, t_eval, nu=0.1, omega=1.0, f=1.2, 
                   truncating_factor = 0.9, plot=False, t0=0):
    """
    Wrapper for the numba ODE solver, adding optional matplotlib plotting.
    """
    dt = t_eval[1] - t_eval[0]
    n_steps = len(t_eval)

    out_theta, out_v = solve_pendulum_numba(theta0, v0, dt, n_steps, nu, omega, f, t0)
    
    out_theta = (out_theta + np.pi) % (2*np.pi) - np.pi
    length = len(out_theta)
    out_theta = out_theta[int(length * truncating_factor):]
    out_v = out_v[int(length*truncating_factor):]

    if plot:
        plt.figure(figsize=(5,5))
        plt.plot(out_theta, out_v, lw=0.5)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\dot{\theta}$')
        plt.title('Phase Space Trajectory')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return out_theta, out_v

@njit
def is_periodic_numba(pi_fraction, thetas, tolerance=1e-3, omega=1.0):
    """
    Checks 2pi periodicity
    """
    step_float = 2 * pi_fraction / omega
    step_int = int(step_float)
    if step_float != step_int:
        raise ValueError("please ensure 2*pi/fraction/omega is an integer")

    return abs(thetas[-step_int - 1] - thetas[-1]) % (2*np.pi) < tolerance

def get_stroboscopic_trajectory(ini_condition,
                                    f=1.2,
                                    divisor=200,
                                    t_max=500.0,
                                    plot = True, **kwargs):
    """
    Returns the 2pi Stroboscopic trajectory. Plotting is optional
    """
    t_eval = np.arange(0, t_max, np.pi/divisor)
    thetas, _ = solve_pendulum(
                        theta0=ini_condition[0],
                        v0=ini_condition[1],
                        t_eval=t_eval,
                        f=f,
                        plot=False, **kwargs)
    
    indexes = np.arange(0, len(thetas), 2*divisor)
    thetas = thetas[indexes]
    ts = t_eval[indexes]

    if (plot):
        plt.figure(figsize=(5,5))
        plt.plot(ts, thetas)
        plt.xlabel("t")
        plt.ylabel(r"$\theta$")
        plt.title("Stroboscopic θ-trajectory")
        #plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return ts, thetas

##############################################################
## Section 1.2: Plot the bifurcation diagram for the autonomous 
# system
##############################################################

def _bifurcation_worker_for_f(f, ini_conditions_f, samples, nu, omega, divisor, t_max, truncating_factor):
    """
    Worker to compute all 2pi stroboscopic points for the given parameter value f
    over 'samples' of initial conditions in ini_conditions_f.
    """
    Fs_list = []
    theta_list = []

    for k in range(samples):
        ini_condition = ini_conditions_f[k]

        _, thetas = get_stroboscopic_trajectory(
            ini_condition=ini_condition,
            nu=nu,
            omega=omega,
            f=f,
            divisor=divisor,
            t_max=t_max,
            truncating_factor=truncating_factor,
            plot=False,
        )

        Fs_list.append(np.full_like(thetas, f, dtype=float))
        theta_list.append(thetas)

    if Fs_list:
        return np.concatenate(Fs_list), np.concatenate(theta_list)
    else:
        return np.empty(0), np.empty(0)

def plot_bifurcation(
        samples=100,
        fmin=-2,
        fmax=2,
        npts=1000,
        nu=0.1,
        omega=1.0,
        divisor=100,
        t_max=2000.0,
        truncating_factor=0.9,
        n_jobs=-1,
        seed=None,
    ):
    """
    Plot the bifurcation diagram. Random initial conditions are chosen at each f
    to ensure more fixed points are found, so that the diagram is more complete. 

    - samples: number of random ICs per F
    - npts: number of F values between fmin and fmax
    - n_jobs: number of parallel workers (-1 = all cores)
    """

    fs = np.linspace(fmin, fmax, npts)
    rng = np.random.default_rng(seed)

    # Pre-generate all random initial conditions: shape (npts, samples, 2)
    ini_conditions = 3 * rng.uniform(-1.0, 1.0, size=(npts, samples, 2))

    # Parallel over F values
    results = Parallel(n_jobs=n_jobs)(
        delayed(_bifurcation_worker_for_f)(
            fs[i],
            ini_conditions[i],
            samples,
            nu,
            omega,
            divisor,
            t_max,
            truncating_factor,
        )
        for i in range(npts)
    )

    # Collect all results
    all_F = []
    all_theta = []
    for F_vals, theta_vals in results:
        all_F.append(F_vals)
        all_theta.append(theta_vals)

    if all_F:
        all_F = np.concatenate(all_F)
        all_theta = np.concatenate(all_theta)
    else:
        all_F = np.array([])
        all_theta = np.array([])

    # Plot
    plt.figure(figsize=(5, 5))
    plt.scatter(all_F, all_theta, s=1, c='black', rasterized=True)
    plt.xlabel("$F$")
    plt.ylabel(r'$\theta_f$')
    plt.title('Bifurcation plot')
    plt.xlim([fmin, fmax])
    plt.tight_layout()
    plt.show()

##############################################################
## Section 1.3: Plot the first fractal
##############################################################

@njit(parallel=True)
def basin_scan_numba(theta_vals, v_vals, dt, n_steps, trunc_start,
                     pi_fraction, omega, nu, f, tol, t0):
    '''
    Scan the 2D grid in the phase space. Returns a mask showing if each initial
    condition is 2pi-periodic or not. Parallelised computation is used.
    '''

    size = theta_vals.size
    mask = np.zeros((size, size), dtype=np.uint8)

    for i in prange(size):
        for j in range(size):
            th0 = theta_vals[i]
            v0 = v_vals[j]

            thetas, vels = solve_pendulum_numba(th0, v0, dt, n_steps, nu, omega, f, t0)

            # remove transient part
            th_tail = thetas[trunc_start:]

            if is_periodic_numba(pi_fraction=pi_fraction, thetas = th_tail, omega = omega, tolerance=tol):
                mask[j, i] = 1

    return mask

def visualise_basin_numba(
        size=200,
        theta_range=(-np.pi, np.pi),
        v_range=(-4, 4),
        omega=1.0,
        nu=0.1,
        f=1.2,
        divisor=1000,
        t_max=1000.0,
        truncating_factor=0.9,
        tol=1e-3,
        show = True,
        t0=0,
        dpi = 200,
        fig_size = (4,3.5)
    ):

    '''
    Wrapper for the 2D grid scan in the phase space.
    '''

    dt = np.pi / divisor
    n_steps = int(t_max / dt)
    trunc_start = int(truncating_factor * n_steps)

    theta_vals = np.linspace(theta_range[0], theta_range[1], size)
    v_vals     = np.linspace(v_range[0], v_range[1], size)

    periodic_mask = basin_scan_numba(
        theta_vals, v_vals,
        dt, n_steps, trunc_start,
        divisor, omega, nu, f, tol, t0
    )

    plt.figure(figsize=(fig_size[0], fig_size[1]), dpi = dpi)
    cmap = ListedColormap(["yellow", "blue"])

    plt.imshow(
        periodic_mask,
        origin='lower',
        extent=[theta_range[0], theta_range[1],
                v_range[0], v_range[1]],
        cmap=cmap,
        aspect='auto'
    )
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\dot{\theta}_0$')
    #plt.title(r"Basin of the period-2$\pi$ attractor")
    plt.tight_layout()

    if show:
        plt.show()

    return theta_vals, v_vals, periodic_mask

@njit(parallel=True)
def basin_cut_scan_numba(x_vals, dt, n_steps, trunc_start,
                         pi_fraction, omega, nu, f, tol, t0,
                         v_cut, v_value, theta_value):
    """
    1D scan along a cut.
    If v_cut=True:  x_vals are theta0 values, v0 = v_value.
    If v_cut=False: x_vals are v0 values,     theta0 = theta_value.

    Returns: mask_1d (uint8), same length as x_vals. 1=periodic, 0=not.
    """
    n = x_vals.size
    mask = np.zeros(n, dtype=np.uint8)

    for k in prange(n):
        if v_cut:
            th0 = x_vals[k]
            v0  = v_value
        else:
            th0 = theta_value
            v0  = x_vals[k]

        thetas, vels = solve_pendulum_numba(th0, v0, dt, n_steps, nu, omega, f, t0)

        th_tail = thetas[trunc_start:]
        if is_periodic_numba(pi_fraction=pi_fraction, thetas=th_tail, omega=omega, tolerance=tol):
            mask[k] = 1

    return mask

def visualise_basin_cut_numba(
    f,
    npts=1000,
    v_cut=True,
    v_value=0.0,
    theta_value=0,
    theta_range=(-np.pi, np.pi),
    v_range=(-4, 4),
    omega=1.0,
    nu=0.1,
    divisor=1000,
    t_max=1000.0,
    truncating_factor=0.9,
    tol=1e-3,
    t0=0,
    show=True,
    dpi = 200,
    fig_size = (5,1.5)
):
    """
    Computes a 1D cut mask and plots a 1D state image of initial conditions
    that are periodic (mask==1).
    """

    dt = np.pi / divisor
    n_steps = int(t_max / dt)
    trunc_start = int(truncating_factor * n_steps)

    if v_cut:
        x_vals = np.linspace(theta_range[0], theta_range[1], npts)
        xlabel = r'$\theta_0$'
        min_val, max_val = theta_range[0], theta_range[1]
    else:
        x_vals = np.linspace(v_range[0], v_range[1], npts)
        xlabel = r'$\dot{\theta}_0$'
        min_val, max_val = v_range[0], v_range[1]

    mask_1d = basin_cut_scan_numba(
        x_vals,
        dt, n_steps, trunc_start,
        divisor, omega, nu, f, tol, t0,
        v_cut, v_value, theta_value
    )

    fontsize = 14

    plt.figure(figsize=(fig_size[0],fig_size[1]), dpi=dpi)
    state_img = np.zeros((32, npts, 3), dtype=float)
    tracked_mask = mask_1d == 1
    state_img[:, tracked_mask] = [0.0, 0.0, 1.0]
    state_img[:, ~tracked_mask] = [1.0, 1.0, 0.0]
    plt.imshow(
        state_img,
        origin='lower',
        extent=[min_val, max_val, 0, 1],
        cmap=None,
        aspect='auto',
        interpolation='nearest'
    )

    plt.ylim([0, 1])
    plt.xlabel(xlabel, fontsize=fontsize)
    ax = plt.gca()

    # set tick locations
    ax.set_xticks([min_val, (min_val + max_val) / 2, max_val])

    def pi_formatter(x, pos):
        tol = 1e-10
        if np.isclose(x, np.pi, atol=tol):
            return r'$\pi$'
        if np.isclose(x, -np.pi, atol=tol):
            return r'$-\pi$'
        return f'{x:.2f}'

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(pi_formatter))
    plt.xticks(fontsize=fontsize)
    plt.yticks([])
    plt.xlim([min_val, max_val])
    plt.tight_layout()
    plt.show()
    if show:
        plt.show()

    return x_vals, mask_1d

##############################################################
## Section 1.4: Implement the uncertainty algorithm on fractal
# 1.
##############################################################

def bootstrapping_slope_confid_interval(slope, log_eps, log_feps, boot_size, repeats = 2000, percent = 95):
    '''
    Compute the confidence interval using bootstrapping
    '''
    
    slopes = []
    indexes = np.arange(len(log_feps))
    
    for i in range(repeats):
        drawn_indexes = np.random.choice(indexes, size = boot_size)
        boot_slope, *_ = linregress(log_eps[drawn_indexes], log_feps[drawn_indexes])
        slopes.append(boot_slope)
    
    # 95% bootstrap percentile CI
    percent_cut = (100 - percent)/2
    lower = np.percentile(slopes, percent_cut)
    upper = np.percentile(slopes, 100 - percent_cut)
    confid_int = max(upper - slope, slope - lower)
    return confid_int

@njit(parallel=True)
def uncertainty_periodicity_scan_numba(theta_vals, v_vals, dt, n_steps, trunc_start,
                                       pi_fraction, omega, nu, f, tol):
    """
    For each initial condition in the input arrays, integrate the pendulum
    and test 2pi-periodicity in parallel.
    Returns mask[k] = 1 if periodic, 0 otherwise.
    """
    n = theta_vals.size
    mask = np.zeros(n, dtype=np.uint8)

    for k in prange(n):
        thetas, _ = solve_pendulum_numba(
            theta_vals[k], v_vals[k], dt, n_steps, nu, omega, f, 0.0
        )
        th_tail = thetas[trunc_start:]

        if is_periodic_numba(
            pi_fraction=pi_fraction,
            thetas=th_tail,
            omega=omega,
            tolerance=tol
        ):
            mask[k] = 1

    return mask

def _update_uncertainty_counts(trials, uncertains, pair_diffs, threshold):
    """
    Update the counts exactly, including the case where the threshold is
    crossed partway through the current batch.
    """
    remaining = threshold - uncertains
    cumulative_hits = np.cumsum(pair_diffs.astype(np.int64))

    if cumulative_hits[-1] < remaining:
        return trials + pair_diffs.size, uncertains + int(cumulative_hits[-1])

    hit_index = int(np.searchsorted(cumulative_hits, remaining, side='left'))
    return trials + hit_index + 1, threshold

def _uncertainty_for_eps(eps, threshold, theta_min, theta_max, vmin, vmax,
                         f, omega, nu, tol, divisor=200, batch_size=1024,
                         rng=None, t_max=500.0, truncating_factor=0.9,
                         max_trials=1_000_000):
    """
    Batched worker for the uncertainty algorithm in the phase space.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = np.pi / divisor
    n_steps = int(t_max / dt)
    trunc_start = int(truncating_factor * n_steps)

    trials = 0
    uncertains = 0
    sampled_pairs = 0

    theta_batch = np.empty(batch_size * 2, dtype=np.float64)
    v_batch = np.empty(batch_size * 2, dtype=np.float64)

    while uncertains < threshold:
        theta1s = rng.uniform(theta_min, theta_max, batch_size)
        v1s = rng.uniform(vmin, vmax, batch_size)
        angles = rng.uniform(0.0, 2.0 * np.pi, batch_size)

        dthetas = eps * np.cos(angles)
        dvs = eps * np.sin(angles)

        theta2s = theta1s + dthetas
        v2s = v1s + dvs

        out_of_bounds = (
            (theta2s < theta_min) | (theta2s > theta_max) |
            (v2s < vmin) | (v2s > vmax)
        )
        theta2s[out_of_bounds] = theta1s[out_of_bounds] - dthetas[out_of_bounds]
        v2s[out_of_bounds] = v1s[out_of_bounds] - dvs[out_of_bounds]

        theta2s = np.clip(theta2s, theta_min, theta_max)
        v2s = np.clip(v2s, vmin, vmax)

        theta_batch[:batch_size] = theta1s
        theta_batch[batch_size:] = theta2s
        v_batch[:batch_size] = v1s
        v_batch[batch_size:] = v2s

        periodic_mask = uncertainty_periodicity_scan_numba(
            theta_batch, v_batch, dt, n_steps, trunc_start,
            divisor, omega, nu, f, tol
        )
        pair_diffs = periodic_mask[:batch_size] != periodic_mask[batch_size:]
        trials, uncertains = _update_uncertainty_counts(
            trials, uncertains, pair_diffs, threshold
        )

        sampled_pairs += batch_size
        if sampled_pairs >= max_trials and uncertains == 0:
            raise ValueError(
                f'No uncertain points found after {sampled_pairs} sampled pairs at eps={eps:.3e}. '
                'The sampled region may lie inside a single basin at this scale.'
            )

    p_hat = threshold / trials
    sigma_p = p_hat * np.sqrt(max(0.0, (1.0 - p_hat)) / threshold)
    return p_hat, sigma_p, trials

def _uncertainty_for_eps_1D(eps, threshold, theta_min, theta_max, vmin, vmax,
                            f, omega, nu, tol, fix_v, theta, v, divisor=200,
                            batch_size=1024, rng=None, t_max=500.0,
                            truncating_factor=0.9, max_trials=1_000_000):
    """
    Batched worker for the 1D uncertainty algorithm in the phase space.
    If fix_v=True:  select random theta in [theta_min, theta_max), v0 = v.
    If fix_v=False: select random v in [v_min, v_max), theta0 = theta.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = np.pi / divisor
    n_steps = int(t_max / dt)
    trunc_start = int(truncating_factor * n_steps)

    trials = 0
    uncertains = 0
    sampled_pairs = 0

    theta_batch = np.empty(batch_size * 2, dtype=np.float64)
    v_batch = np.empty(batch_size * 2, dtype=np.float64)

    while uncertains < threshold:
        if fix_v:
            theta1s = rng.uniform(theta_min, theta_max, batch_size)
            theta2s = np.where(theta1s + eps <= theta_max, theta1s + eps, theta1s - eps)
            theta2s = np.clip(theta2s, theta_min, theta_max)

            v1s = np.full(batch_size, v, dtype=np.float64)
            v2s = v1s.copy()
        else:
            v1s = rng.uniform(vmin, vmax, batch_size)
            v2s = np.where(v1s + eps <= vmax, v1s + eps, v1s - eps)
            v2s = np.clip(v2s, vmin, vmax)

            theta1s = np.full(batch_size, theta, dtype=np.float64)
            theta2s = theta1s.copy()

        theta_batch[:batch_size] = theta1s
        theta_batch[batch_size:] = theta2s
        v_batch[:batch_size] = v1s
        v_batch[batch_size:] = v2s

        periodic_mask = uncertainty_periodicity_scan_numba(
            theta_batch, v_batch, dt, n_steps, trunc_start,
            divisor, omega, nu, f, tol
        )
        pair_diffs = periodic_mask[:batch_size] != periodic_mask[batch_size:]
        trials, uncertains = _update_uncertainty_counts(
            trials, uncertains, pair_diffs, threshold
        )

        sampled_pairs += batch_size
        if sampled_pairs >= max_trials and uncertains == 0:
            raise ValueError(
                f'No uncertain points found after {sampled_pairs} sampled pairs at eps={eps:.3e}. '
                'This 1D cut may stay inside a single basin at this scale.'
            )

    p_hat = threshold / trials
    sigma_p = p_hat * np.sqrt(max(0.0, (1.0 - p_hat)) / threshold)
    return p_hat, sigma_p, trials

def loglog_linear_fit(
    epsilons,
    feps,
    feps_err=None,
    fit_truncation_order=-np.inf,
    num_eps=None,
    plot=False,
    title='',
    return_all_info=False,
):
    """
    The log-log linear fit for the uncertainty algorithms.
    """

    # Mask in log10 space
    log_eps_all = np.log10(epsilons)
    mask = log_eps_all < fit_truncation_order

    log_eps  = log_eps_all[mask]
    log_feps = np.log10(feps)[mask]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(log_eps, log_feps)

    if num_eps is None:
        num_eps = log_eps.size

    confid_int = bootstrapping_slope_confid_interval(
        slope=slope,
        log_eps=log_eps,
        log_feps=log_feps,
        boot_size=int(num_eps),
    )

    fit_line = 10**intercept * epsilons**slope

    # Plot
    if plot:
        plt.figure(figsize=(6, 5))

        if feps_err is not None:
            plt.errorbar(
                epsilons, feps, yerr=feps_err,
                fmt='o', capsize=3, label='Data'
            )
        else:
            plt.loglog(epsilons, feps, 'o', label='Data')

        plt.loglog(
            epsilons, fit_line,
            label=f'Fit: α = {slope:.3f} ± {confid_int:.3f}'
        )

        plt.xlabel('$\\varepsilon$')
        plt.ylabel('$f(\\varepsilon)$')
        if title:
            plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    if return_all_info:
        return epsilons, feps, feps_err, fit_line, slope, confid_int
    else:
        return slope

def uncertainty_algorithm(theta_min=-np.pi, theta_max=np.pi,
                          vmin=-3, vmax=3, threshold=1000, plot=True,
                          verbose=True, min_eps=1e-10, max_eps=1e-6, num_eps=20,
                          return_all_info=False, f=1.2, omega=1, nu=0.1, tol=1e-3,
                          n_jobs=1, fit_truncation_order = 0, divisor = 200,
                          batch_size=1024, t_max=500.0, truncating_factor=0.9,
                          base_seed=0, max_trials=1_000_000, **kwargs):

    '''
    Wrapper for the uncertainty algorithm in the phase space.
    Uses batched initial conditions and a vectorised periodicity scan.
    '''

    epsilons = np.geomspace(min_eps, max_eps, num_eps)
    feps = []
    feps_err = []

    if verbose:
        print(
            f"Running uncertainty algorithm with batch_size={batch_size} "
            f"(n_jobs={n_jobs}, divisor={divisor})..."
        )
        iterator = tqdm(
            enumerate(epsilons),
            total=len(epsilons),
            desc="Sweeping the phase space: "
        )
    else:
        iterator = enumerate(epsilons)

    for i, eps in iterator:
        rng = np.random.default_rng(base_seed + i)
        p_hat, sigma_p, _ = _uncertainty_for_eps(
            eps=eps,
            threshold=threshold,
            theta_min=theta_min,
            theta_max=theta_max,
            vmin=vmin,
            vmax=vmax,
            f=f,
            omega=omega,
            nu=nu,
            tol=tol,
            divisor=divisor,
            batch_size=batch_size,
            rng=rng,
            t_max=t_max,
            truncating_factor=truncating_factor,
            max_trials=max_trials
        )
        feps.append(p_hat)
        feps_err.append(sigma_p)

    return loglog_linear_fit(
        epsilons,
        feps,
        feps_err=feps_err,
        fit_truncation_order=fit_truncation_order,
        plot=plot,
        title='f(蔚) vs 蔚 for the first fractal',
        return_all_info=return_all_info,
    )

def uncertainty_algorithm_1D(theta_min=-np.pi, theta_max=np.pi,
                          vmin=-3, vmax=3, threshold=1000, plot=True,
                          verbose=True, min_eps=1e-10, max_eps=1e-6, num_eps=20,
                          return_all_info=False, f=1.2, omega=1, nu=0.1, tol=1e-3,
                          n_jobs=1, fix_v = True, theta = 0, v = 0, fit_truncation_order = 0,
                          divisor = 200, batch_size=1024, t_max=500.0,
                          truncating_factor=0.9, base_seed=0,
                          max_trials=1_000_000, **kwargs):

    '''
    Wrapper for the 1D uncertainty algorithm in the phase space.
    Uses batched initial conditions and a vectorised periodicity scan.
    '''

    epsilons = np.geomspace(min_eps, max_eps, num_eps)
    feps = []
    feps_err = []
    cut_direction = 'theta' if fix_v else 'v'

    if verbose:
        print(
            f"Running 1D uncertainty algorithm with batch_size={batch_size}, "
            f"cut_direction={cut_direction} (n_jobs={n_jobs}, divisor={divisor})..."
        )
        iterator = tqdm(
            enumerate(epsilons),
            total=len(epsilons),
            desc=f"{cut_direction}-cut sweep"
        )
    else:
        iterator = enumerate(epsilons)

    for i, eps in iterator:
        rng = np.random.default_rng(base_seed + i)
        p_hat, sigma_p, _ = _uncertainty_for_eps_1D(
            eps=eps,
            threshold=threshold,
            theta_min=theta_min,
            theta_max=theta_max,
            vmin=vmin,
            vmax=vmax,
            f=f,
            omega=omega,
            nu=nu,
            tol=tol,
            fix_v=fix_v,
            theta=theta,
            v=v,
            divisor=divisor,
            batch_size=batch_size,
            rng=rng,
            t_max=t_max,
            truncating_factor=truncating_factor,
            max_trials=max_trials
        )
        feps.append(p_hat)
        feps_err.append(sigma_p)

    feps = np.asarray(feps)
    feps_err = np.asarray(feps_err)

    return loglog_linear_fit(
        epsilons,
        feps,
        feps_err=feps_err,
        fit_truncation_order=fit_truncation_order,
        plot=plot,
        title='f(ε) vs ε for the first fractal',
        return_all_info=return_all_info,
    )

def alpha_vs_F(fmin = 0.9, fmax = 1.5, npts = 10, **kwargs):
    '''
    Plot the codimension over a range of forcing parameters.
    '''

    fs = np.linspace(fmin, fmax, npts)
    alphas = []

    iterator = tqdm(fs, desc="Progress: ")
    for f in iterator:
        alpha = uncertainty_algorithm_1D(f=f, plot = False, **kwargs)
        alphas.append(alpha)

    plt.figure(figsize=(6, 5))
    plt.plot(fs, alphas)
    plt.xlabel(r'$f$')
    plt.ylabel(r'$\alpha$')
    plt.title(r'$\alpha$ vs. $F$ for the first fractal')
    plt.tight_layout()
    plt.grid()
    plt.show()

##############################################################
## Section 2.1: Plot the second fractal
##############################################################

@njit(parallel=True)
def f_basin_scan_numba(theta_vals, f_vals, dt, n_steps, trunc_start,
                     pi_fraction, omega, nu, v0, tol):
    '''
    Scan the 2D grid in the parameter space of f. Returns a mask showing 
    if each initial condition is 2pi-periodic or not. Parallelised 
    computation is used.
    '''
    
    size = theta_vals.size
    mask = np.zeros((size, size), dtype=np.uint8)

    for i in prange(size):
        for j in range(size):
            th0 = theta_vals[i]
            f = f_vals[j]
            t0 = 0

            thetas, vels = solve_pendulum_numba(th0, v0, dt, n_steps, nu, omega, f, t0)

            # remove transient part
            th_tail = thetas[trunc_start:]

            if is_periodic_numba(pi_fraction=pi_fraction, thetas = th_tail, omega = omega, tolerance=tol):
                mask[j, i] = 1

    return mask

def get_fixed_points_F(fmin = 0, fmax = 1.5, npts = 500, max_tries = 50, t_max = 500, divisor = 200, plot = True):
    '''
    For a grid of f, find a stable fixed point for each f.
    '''
    
    fs = np.linspace(fmin, fmax, npts)
    fixed_pts = []
    t_eval = np.arange(0, t_max, np.pi/divisor)

    for f in fs:
        fixed_pt = find_random_periodic_ic(
            f=f, t_eval=t_eval, divisor=divisor,max_tries=max_tries, 
            rng=None, return_final=True, message=False
        )

        fixed_pts.append(fixed_pt)

    if (plot):
        plt.figure(figsize=(6, 5))
        plt.plot(fs, fixed_pts)
        plt.xlabel('f')
        plt.ylabel(r'Fixed points')
        plt.show()

    return fixed_pts

def visualise_second_fractal_numba(
        fmin, fmax,
        size=200,
        omega=1.0,
        nu=0.1,
        divisor=100,
        t_max=500.0,
        truncating_factor=0.9,
        tol=1e-3,
        theta_min = -np.pi,
        theta_max = np.pi,
        v0=0
    ):

    '''
    Visualise the fractal in the theta-F space.
    '''

    f_range=(fmin, fmax)
    theta_range=(theta_min, theta_max)

    dt = np.pi / divisor
    n_steps = int(t_max / dt)
    trunc_start = int(truncating_factor * n_steps)

    theta_vals = np.linspace(theta_range[0], theta_range[1], size)
    f_vals     = np.linspace(f_range[0], f_range[1], size)

    periodic_mask = f_basin_scan_numba(
        theta_vals, f_vals,
        dt, n_steps, trunc_start,
        divisor, omega, nu, v0, tol
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(
        periodic_mask,
        origin='lower',
        extent=[theta_range[0], theta_range[1],
                f_range[0], f_range[1]],
        cmap='binary',
        aspect='auto'
    )
    
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$f$')
    plt.title(r"Basin of the period-2$\pi$ attractor")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #return theta_vals, f_vals, periodic_mask

@njit(parallel=True)
def vertical_cut_scan_numba(theta0, v0, fs, dt, n_steps,
                            nu, omega, pi_fraction, tolerance):
    """
    For each f in fs, integrate the pendulum and test periodicity.
    Returns mask[k] = 1 if periodic, 0 otherwise.
    """
    n = fs.size
    mask = np.zeros(n, dtype=np.uint8)
    t0=0.0

    for k in prange(n):
        f = fs[k]
        thetas, vels = solve_pendulum_numba(theta0, v0, dt, n_steps, nu, omega, f, t0)

        if is_periodic_numba(pi_fraction, thetas, tolerance, omega):
            mask[k] = 1
        else:
            mask[k] = 0

    return mask

def find_random_periodic_ic(
        f, t_eval,
        nu=0.1, omega=1.0,
        divisor=100, tol=1e-3,
        truncating_factor=0.9,
        max_tries=100,
        rng=None, return_final = False, message = True
    ):
    """
    Randomly search for a 2π-periodic initial condition at forcing f.

    theta0 ~ U(-pi, pi), v0 ~ U(-2, 2).
    Integrates using solve_pendulum and checks is_periodic_numba.
    Returns (theta0, v0) if found, otherwise raises RuntimeError.
    """

    if rng is None:
        rng = np.random.default_rng()

    found = False
    theta0 = 0.0
    v0 = 0.0

    for _ in range(max_tries):
        theta0_candidate = rng.uniform(-np.pi, np.pi)
        v0_candidate = rng.uniform(-2.0, 2.0)

        thetas, _ = solve_pendulum(
            theta0=theta0_candidate,
            v0=v0_candidate,
            t_eval=t_eval,
            nu=nu,
            omega=omega,
            f=f,
            truncating_factor=truncating_factor,
            plot=False,
        )

        if is_periodic_numba(pi_fraction=divisor,
                             thetas=thetas,
                             tolerance=tol,
                             omega=omega):
            theta0 = theta0_candidate
            v0 = v0_candidate
            found = True
            break

    if not found:
        raise RuntimeError("Could not find a 2π-periodic initial condition at fmin within max_tries.")

    if (message):
        print(f"Using periodic IC at f={f}: theta0={theta0:.6f}, v0={v0:.6f}")

    if (not return_final):
        return theta0, v0
    else:
        return thetas[-1]

def visualise_vertical_cut(
        theta0, v0,
        fmin=1.2,
        fmax=1.3,
        npts=500,
        nu=0.1,
        omega=1.0,
        divisor=100,
        t_max=500.0,
        tol=1e-3,
        rng=None,
        fig_size =(4, 1.5),
        dpi = 200
    ):
    """
    Visualise the fractal in F space (the second fractal).
    """

    if rng is None:
        rng = np.random.default_rng()

    # --- time grid ---
    dt = np.pi / divisor
    t_eval = np.arange(0, t_max, dt)
    n_steps = len(t_eval)

    # --- 2. build F grid ---
    fs = np.linspace(fmin, fmax, npts)

    # --- 3. use numba-accelerated scan with this IC ---
    mask = vertical_cut_scan_numba(
        theta0=theta0,
        v0=v0,
        fs=fs,
        dt=dt,
        n_steps=n_steps,
        nu=nu,
        omega=omega,
        pi_fraction=divisor,
        tolerance=tol
    )

    plt.figure(figsize=(fig_size[0], fig_size[1]), dpi = dpi)

    fontsize = 14

    state_img = np.zeros((32, npts, 3), dtype=float)
    periodic_mask = mask == 1
    state_img[:, periodic_mask] = [0.0, 0.0, 1.0]
    state_img[:, ~periodic_mask] = [1.0, 1.0, 0.0]
    plt.imshow(
        state_img,
        origin='lower',
        extent=[fmin, fmax, 0, 1],
        cmap=None,
        aspect='auto',
        interpolation='nearest'
    )

    plt.ylim([0, 1])
    plt.xlabel("$\lambda_+$", fontsize=fontsize)
    ax = plt.gca()
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    plt.xticks([fmin, (fmin+fmax)/2, fmax], fontsize=fontsize)
    plt.yticks([])
    # plt.title(rf'Fractal 2')
    plt.xlim([fmin, fmax])
    plt.tight_layout()
    plt.show()

##############################################################
## Section 2.2: Implement the uncertainty algorithm on Fractal
# 2.
##############################################################

def _uncertainty_for_eps_F(eps, threshold, theta0, v0, fmin, fmax, omega, nu, tol,
                           divisor=200, batch_size=1024, rng=None, t_max=500.0,
                           max_trials=1_000_000):
    """
    Batched worker for the uncertainty algorithm of the second fractal.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = np.pi / divisor
    n_steps = int(t_max / dt)

    trials = 0
    uncertains = 0
    sampled_pairs = 0
    fs = np.empty(batch_size * 2, dtype=np.float64)

    while uncertains < threshold:
        f1s = rng.uniform(fmin, fmax, batch_size)
        f2s = np.where(f1s + eps <= fmax, f1s + eps, f1s - eps)
        f2s = np.clip(f2s, fmin, fmax)

        fs[:batch_size] = f1s
        fs[batch_size:] = f2s

        periodic_mask = vertical_cut_scan_numba(
            theta0=theta0,
            v0=v0,
            fs=fs,
            dt=dt,
            n_steps=n_steps,
            nu=nu,
            omega=omega,
            pi_fraction=divisor,
            tolerance=tol
        )
        pair_diffs = periodic_mask[:batch_size] != periodic_mask[batch_size:]
        trials, uncertains = _update_uncertainty_counts(
            trials, uncertains, pair_diffs, threshold
        )

        sampled_pairs += batch_size
        if sampled_pairs >= max_trials and uncertains == 0:
            raise ValueError(
                f'No uncertain points found after {sampled_pairs} sampled pairs at eps={eps:.3e}. '
                'The sampled forcing interval may lie inside a single basin at this scale.'
            )

    p_hat = threshold / trials
    sigma_p = p_hat * np.sqrt(max(0.0, (1.0 - p_hat)) / threshold)
    return p_hat, sigma_p, trials

def uncertainty_algorithm_F(theta0, v0, fmin = 0, fmax = 1.8, threshold=1000, plot=True,
                          verbose=True, min_eps=1e-10, max_eps=1e-6, num_eps=20,
                          return_all_info=False, f=1.2, omega=1, nu=0.1, tol=1e-3,
                          n_jobs=1, fit_truncation_order = 0, divisor=200,
                          batch_size=1024, t_max=500.0, base_seed=0,
                          max_trials=1_000_000, **kwargs):

    '''
    Implement the uncertainty algorithm for the second fractal.
    Uses batched forcing values and a vectorised periodicity scan.
    '''

    epsilons = np.geomspace(min_eps, max_eps, num_eps)
    feps = []
    feps_err = []

    if verbose:
        print(
            f"Running second-fractal uncertainty algorithm with batch_size={batch_size} "
            f"(divisor={divisor})..."
        )
        iterator = tqdm(
            enumerate(epsilons),
            total=len(epsilons),
            desc="fractal-2 eps sweep"
        )
    else:
        iterator = enumerate(epsilons)

    for i, eps in iterator:
        rng = np.random.default_rng(base_seed + i)
        p_hat, sigma_p, _ = _uncertainty_for_eps_F(
            eps=eps,
            threshold=threshold,
            theta0=theta0,
            v0=v0,
            fmin=fmin,
            fmax=fmax,
            omega=omega,
            nu=nu,
            tol=tol,
            divisor=divisor,
            batch_size=batch_size,
            rng=rng,
            t_max=t_max,
            max_trials=max_trials
        )
        feps.append(p_hat)
        feps_err.append(sigma_p)

    return loglog_linear_fit(
        epsilons,
        feps,
        feps_err=feps_err,
        fit_truncation_order=fit_truncation_order,
        plot=plot,
        title='f(ε) vs ε for the second fractal',
        return_all_info=return_all_info,
    )

##############################################################
## Section 3.1: Solving the damped driven pendulum for a time-
## varying f (non-autonomous system)
##############################################################

@njit
def dfdt(f, r, f_final):
    """
    df/dt = r if f < f_final
            0 otherwise
    """
    if f < f_final:
        return r
    else:
        return 0.0

@njit
def rk4_step_with_drift(theta, v, f, t, dt, nu, omega, r, f_final):
    # k1
    k1_theta = v
    k1_v = f*math.cos(t) - nu*v - math.sin(theta)
    k1_f = dfdt(f, r, f_final)

    # k2
    theta2 = theta + 0.5*dt*k1_theta
    v2 = v + 0.5*dt*k1_v
    f2 = f + 0.5*dt*k1_f
    k2_theta = v2
    k2_v = f2*math.cos(t + 0.5*dt) - nu*v2 - math.sin(theta2)
    k2_f = dfdt(f2, r, f_final)

    # k3
    theta3 = theta + 0.5*dt*k2_theta
    v3 = v + 0.5*dt*k2_v
    f3 = f + 0.5*dt*k2_f
    k3_theta = v3
    k3_v = f3*math.cos(t + 0.5*dt) - nu*v3 - math.sin(theta3)
    k3_f = dfdt(f3, r, f_final)

    # k4
    theta4 = theta + dt*k3_theta
    v4 = v + dt*k3_v
    f4 = f + dt*k3_f
    k4_theta = v4
    k4_v = f4*math.cos(t + dt) - nu*v4 - math.sin(theta4)
    k4_f = dfdt(f4, r, f_final)

    theta_new = theta + (dt/6.0)*(k1_theta + 2.0*k2_theta + 2.0*k3_theta + k4_theta)
    v_new     = v     + (dt/6.0)*(k1_v     + 2.0*k2_v     + 2.0*k3_v     + k4_v)
    f_new     = f     + (dt/6.0)*(k1_f     + 2.0*k2_f     + 2.0*k3_f     + k4_f)

    return theta_new, v_new, f_new

@njit
def solve_pendulum_with_drift_numba(theta0, v0, dt, n_steps, nu, omega, f0, r, f_final, t0):
    theta = theta0
    v = v0
    f = f0        
    t = t0

    out_theta = np.empty(n_steps)
    out_v     = np.empty(n_steps)
    out_f     = np.empty(n_steps)

    for i in range(n_steps):
        out_theta[i] = theta
        out_v[i]     = v
        out_f[i]     = f

        theta, v, f = rk4_step_with_drift(theta, v, f, t, dt, nu, omega, r, f_final)
        t += dt

    return out_theta, out_v, out_f

def solve_pendulum_with_drift(t_eval, r, f0, f_final, theta0, v0, nu=0.1,omega=1.0, plot=False, t0=0, mod2pi = True):
    """
    Python wrapper around solve_pendulum_with_drift_numba.

    - Returns (theta, v, f) time series.
    - If plot is True, plots theta vs time.
    """

    dt = t_eval[1] - t_eval[0]
    n_steps = len(t_eval)
    
    thetas, vels, fs = solve_pendulum_with_drift_numba(
        theta0,
        v0,
        dt,
        n_steps,
        nu,
        omega,
        f0,
        r,
        f_final,
        t0
    )

    if (mod2pi):
        thetas = (thetas + np.pi) % (2*np.pi) - np.pi

    if plot:
        plt.figure(figsize=(5, 4))
        plt.plot(dt * np.arange(1, n_steps+1), thetas, lw=0.7)
        plt.xlabel(r"t")
        plt.ylabel(r"$\theta$")
        plt.title("Drifting pendulum: $\\theta$ vs $t$")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return thetas, vels, fs

def get_stroboscopic_trajectory_with_drift(
        rate,
        omega=1.0,
        nu=0.1,
        f0=0,        
        f_final=1.2,    
        divisor=200,
        t_max=500.0,
        plot=True,
        theta0 = None, v0 = None):
    """
    Returns the 2pi Stroboscopic trajectory. Plotting is optional.
    Only works for single divisor (small rates)
    """

    t_eval = np.arange(0, t_max, np.pi/divisor)

    if (theta0 == None or v0 == None):
        theta0, v0 = find_random_periodic_ic(
                f=f0,
                t_eval=t_eval,
                nu=nu,
                omega=omega,
                divisor=divisor,
                tol=1e-3,
                truncating_factor=0.9,
                max_tries=100,
                rng=None
        )

    # solve drifting ODE
    thetas, vs, fs = solve_pendulum_with_drift(
        theta0=theta0,
        v0=v0,
        t_eval=t_eval,
        nu=nu,
        omega=omega,
        f0=f0,
        r=rate,
        f_final=f_final,
        plot=False
    )

    # stroboscopic sampling every forcing period 2π
    indexes = np.arange(0, len(thetas), 2*divisor)

    thetas_sampled = thetas[indexes]
    ts_sampled = t_eval[indexes]
    vs_sampled = vs[indexes]

    # plot
    if plot:
        plt.figure(figsize=(5,5))
        plt.plot(ts_sampled, thetas_sampled)
        plt.xlabel("t")
        plt.ylabel(r"$\theta$")
        plt.title("Stroboscopic θ-trajectory (with drift)")
        plt.tight_layout()
        plt.show()

    return ts_sampled, thetas_sampled, vs_sampled

def solve_pendulum_with_drift_all_rates(rate, theta0, v0, f0, f_final, omega=1, nu=0.1, divisor = 200, settling_time=500, 
                                        large_rate_multiple=1000, large_rate_limit=1, drift_time_multiple = 1.1):
    '''
    For large rates, we need to make dt finer so that the integration will not overshoot the forcing too much. Hence, during the 
    very short time of parameter drift, we use a very large divisor (large_rate_multiple * rate) to make dt very fine. Then, we 
    use the user-provided divisor, which is coarser, to save computational time.

    drift_time_multiple > 1 integrate a bit longer, ensuring the integration brings f over f_final (with a bit of overshoot).
    If it's equal to 1, the last f value will be a bit below f_final. Both cases are fine as long as the error of integration
    is small. 
    '''

    if (rate == 0):
        times = np.arange(0, settling_time, np.pi/divisor)
        thetas, vels, fs = solve_pendulum_with_drift(
            t_eval=times, r=0, f0=f0, f_final=f0,
            theta0=theta0, v0=v0, plot=False, omega=omega, nu=nu
        )
        return times, thetas, vels, fs
    elif (0 < rate <= large_rate_limit):
        t_max = (f_final - f0)/rate + settling_time
        times = np.arange(0, t_max, np.pi/divisor)
        thetas, vels, fs = solve_pendulum_with_drift(
            t_eval=times, r=rate, f0=f0, f_final=f_final,
            theta0=theta0, v0=v0, plot=False, omega=omega, nu=nu
        )
        return times, thetas, vels, fs
    elif (rate > large_rate_limit):
        divisor1 = max(divisor, large_rate_multiple * rate)

        if (drift_time_multiple < 1):
            drift_time_multiple = 1
                                  
        t_max = (f_final - f0)/rate * drift_time_multiple 
        times = np.arange(0, t_max, np.pi/divisor1)
        thetas, vels, fs = solve_pendulum_with_drift(
            t_eval=times, r=rate, f0=f0, f_final=f_final,
            theta0=theta0, v0=v0, plot=False, omega=omega, nu=nu
        )
        
        times1 = np.arange(times[-1], times[-1] + settling_time, np.pi/divisor)
        thetas1, vels1, fs1 = solve_pendulum_with_drift(
            t_eval=times1, r=0, f0=fs[-1], f_final=fs[-1],
            theta0=thetas[-1], v0=vels[-1], plot=False,
            omega=omega, nu=nu
        )

        times = np.concatenate((times, times1))
        thetas = np.concatenate((thetas, thetas1))
        vels = np.concatenate((vels, vels1))
        fs =  np.concatenate((fs, fs1))

        return times, thetas, vels, fs
    
    else:
        raise ValueError("Rate must be non-negative.")

##############################################################
## Section 3.2: Plot the third fractal
##############################################################

def _third_fractal_worker(rate, theta0, v0, f0, f_final, omega, nu, tol, divisor, settling_time, large_rate_multiple, large_rate_limit):
    """
    Worker to visualise the third fractal using parallelised computation
    """

    fs_last = None

    if (0 < rate <= large_rate_limit):
        t_max = (f_final - f0)/rate + settling_time
        _, thetas, _ = get_stroboscopic_trajectory_with_drift(
            omega=omega,
            nu=nu,
            f0=f0,
            rate=rate,
            f_final=f_final,
            divisor=divisor,
            t_max=t_max,
            theta0=theta0,
            v0=v0,
            plot=False
        )

        is_periodic = abs(thetas[-1] - thetas[-2]) < tol

    elif (rate > large_rate_limit):
        divisor1 = max(divisor, large_rate_multiple * rate)
        t_max = (f_final - f0)/rate * 1.1
        t_eval = np.arange(0, t_max, np.pi/divisor1)
        thetas, vels, fs = solve_pendulum_with_drift(t_eval=t_eval, r = rate, f0 = f0, f_final = f_final, 
                                                    theta0=theta0, v0=v0, plot=False)
        
        fs_last = float(fs[-1])

        _, thetas = get_stroboscopic_trajectory(ini_condition=[thetas[-1], vels[-1]], f=fs[-1],   
            divisor=divisor, t_max=settling_time + t_eval[-1], plot = False, t0=t_eval[-1])

        is_periodic = abs(thetas[-1] - thetas[-2]) < tol

    else: # rate = 0
        _, thetas = get_stroboscopic_trajectory(ini_condition=[theta0, v0], f=f0,   
            divisor=divisor, t_max=settling_time, plot = False)

        is_periodic = abs(thetas[-1] - thetas[-2]) < tol

    return rate, is_periodic, fs_last

def visualise_third_fractal(rate_min=0, rate_max=2, npts=1000,
                            f0=1.1, f_final=1.3, omega=1.0, nu=0.1,
                            tol=1e-3, max_tries=100, divisor = 100,
                            n_jobs=-1, theta0 = None, v0 = None, 
                            settling_time = 1000, large_rate_multiple = 1000,
                            large_rate_limit = 1, fig_size =(5, 1.5), dpi = 200):
    
    '''
    Wrapper to visualise the third fractal.
    '''

    rates = np.linspace(rate_min, rate_max, npts)
    t_eval = np.arange(0, 2000, np.pi/divisor)

    if (theta0 == None or v0 == None):
        theta0, v0 = find_random_periodic_ic(
            f=f0,
            t_eval=t_eval,
            nu=nu,
            omega=omega,
            divisor=100,
            tol=tol,
            truncating_factor=0.9,
            max_tries=max_tries,
            rng=None
        )

    
    # run all rates in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_third_fractal_worker)(
            rate, theta0, v0, f0, f_final, omega, nu, tol, divisor, settling_time, large_rate_multiple, large_rate_limit
        )
        for rate in rates
    )

    # print all fs_last values that exist
    # for rate, is_per, fs_last in results:
    #     if fs_last is not None:
    #         print(f"rate={rate:.6g}, fs[-1]={fs_last:.17g}")

    periodic_mask = np.asarray([is_per for (_, is_per, _) in results], dtype=bool)
    periodic_rates = rates[periodic_mask]
    non_periodic_rates = rates[~periodic_mask]

    print(min(non_periodic_rates), max(periodic_rates))

    font_size = 14

    plt.figure(figsize=(fig_size[0], fig_size[1]), dpi = dpi)

    state_img = np.zeros((32, npts, 3), dtype=float)
    state_img[:, periodic_mask] = [0.0, 0.0, 1.0]
    state_img[:, ~periodic_mask] = [1.0, 1.0, 0.0]
    plt.imshow(
        state_img,
        origin='lower',
        extent=[rate_min, rate_max, 0, 1],
        cmap=None,
        aspect='auto',
        interpolation='nearest'
    )

    plt.ylim([0, 1])
    plt.xlabel("$r$", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks([])
    #plt.title(rf'Fractal 3; $F_0={f0}$; $F_f = {f_final}$; dt = {np.pi/divisor:.5f}')
    plt.xlim([rate_min, rate_max])
    plt.tight_layout()
    plt.show()

##############################################################
## Section 3.3: Implement the uncertainty algorithm on Fractal
# 3.
##############################################################

def _rate_periodicity_worker(rate, theta0, v0, f0, f_final, omega, nu, tol, divisor,
                             settling_time, large_rate_multiple, large_rate_limit):
    """
    Classify one rate as periodic/non-periodic using the adaptive drift solver.
    """
    _, thetas, _, _ = solve_pendulum_with_drift_all_rates(
        rate=rate,
        theta0=theta0,
        v0=v0,
        f0=f0,
        f_final=f_final,
        omega=omega,
        nu=nu,
        divisor=divisor,
        settling_time=settling_time,
        large_rate_multiple=large_rate_multiple,
        large_rate_limit=large_rate_limit
    )
    return is_periodic_numba(
        pi_fraction=divisor,
        thetas=thetas,
        omega=omega,
        tolerance=tol
    )

def _uncertainty_for_eps_rate(eps, threshold, theta0, v0, f0, f_final, rate_min, rate_max,
                              omega, nu, tol, divisor, batch_size=2048, rng=None,
                              max_trials=1_000_000, n_jobs=-1, settling_time=500,
                              large_rate_multiple=1000, large_rate_limit=1,
                              prefer='processes'):
    """
    Batched worker for the uncertainty algorithm of the third fractal.
    """
    if rng is None:
        rng = np.random.default_rng()

    trials = 0
    uncertains = 0
    sampled_pairs = 0

    while uncertains < threshold:
        rate1s = rng.uniform(rate_min, rate_max, batch_size)
        rate2s = np.where(rate1s + eps <= rate_max, rate1s + eps, rate1s - eps)
        rate2s = np.clip(rate2s, rate_min, rate_max)
        rates = np.concatenate((rate1s, rate2s))

        periodic_mask = np.asarray(
            Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(_rate_periodicity_worker)(
                    rate, theta0, v0, f0, f_final, omega, nu, tol, divisor,
                    settling_time, large_rate_multiple, large_rate_limit
                )
                for rate in rates
            ),
            dtype=bool
        )

        pair_diffs = periodic_mask[:batch_size] != periodic_mask[batch_size:]
        trials, uncertains = _update_uncertainty_counts(
            trials, uncertains, pair_diffs, threshold
        )

        sampled_pairs += batch_size
        if sampled_pairs >= max_trials and uncertains == 0:
            raise ValueError(
                f'No uncertain points found after {sampled_pairs} sampled pairs at eps={eps:.3e}. '
                'The sampled rate interval may lie inside a single basin at this scale.'
            )

    p_hat = threshold / trials
    sigma_p = p_hat * np.sqrt(max(0.0, (1.0 - p_hat)) / threshold)
    return p_hat, sigma_p, trials

def uncertainty_algorithm_rate(theta0, v0, f0 = 0, f_final = 1.3, rate_min = 0, rate_max = 1,
    threshold=1000, plot=True, verbose=True, min_eps=1e-10, max_eps=1e-6, num_eps=20,
    return_all_info=False, f=1.2, omega=1, nu=0.1, tol=1e-3, n_jobs=-1, fit_truncation_order = 0,
    divisor = 200, batch_size=2048, base_seed=0, max_trials=1_000_000,
    settling_time=500, large_rate_multiple=1000, large_rate_limit=1,
    prefer='processes', **kwargs):

    """
    Implement the uncertainty algorithm for the third fractal.
    Uses batched rate samples and parallel adaptive drift solves.
    """

    epsilons = np.geomspace(min_eps, max_eps, num_eps)
    feps = []
    feps_err = []

    if verbose:
        print(
            f"Running third-fractal uncertainty algorithm with batch_size={batch_size} "
            f"(n_jobs={n_jobs}, divisor={divisor})..."
        )
        iterator = tqdm(
            enumerate(epsilons),
            total=len(epsilons),
            desc="fractal-3 eps sweep"
        )
    else:
        iterator = enumerate(epsilons)

    for i, eps in iterator:
        rng = np.random.default_rng(base_seed + i)
        p_hat, sigma_p, _ = _uncertainty_for_eps_rate(
            eps=eps,
            threshold=threshold,
            theta0=theta0,
            v0=v0,
            f0=f0,
            f_final=f_final,
            rate_min=rate_min,
            rate_max=rate_max,
            omega=omega,
            nu=nu,
            tol=tol,
            divisor=divisor,
            batch_size=batch_size,
            rng=rng,
            max_trials=max_trials,
            n_jobs=n_jobs,
            settling_time=settling_time,
            large_rate_multiple=large_rate_multiple,
            large_rate_limit=large_rate_limit,
            prefer=prefer
        )
        feps.append(p_hat)
        feps_err.append(sigma_p)

    return loglog_linear_fit(
        epsilons,
        feps,
        feps_err=feps_err,
        fit_truncation_order=fit_truncation_order,
        plot=plot,
        title='f(ε) vs ε for the third fractal',
        return_all_info=return_all_info,
    )

##############################################################
## Section 4.1: Visual animation of the autonomous system
##############################################################    

def run_with_gui_backend(func, *args, backend="TkAgg", **kwargs):
    '''
    Since the animation uses a different backend (TkAgg), need to switch back
    to the original backend after the animation.
    '''
    old_backend = matplotlib.get_backend()
    try:
        matplotlib.use(backend)
        return func(*args, **kwargs)
    finally:
        plt.close("all")
        matplotlib.use(old_backend)

def animate_pendulum(
    theta0, v0, t_eval,
    nu=0.1, omega=1.0, f=1.2,
    L=1.0,
    speed=1.0,
    trail=0,
    divisor=200,
    show_strobe=True,       
    strobe_size=18,
    delay = 0
):
    '''
    Animate the pendulum with fixed forcing (autonomous system)
    '''

    thetas, _ = solve_pendulum(
        theta0=theta0, v0=v0, t_eval=t_eval,
        nu=nu, omega=omega, f=f,
        truncating_factor=0, plot=False)

    if (speed < 1):
        print('Speed is less than 1, setting speed = 1')
        speed = 1
    
    frames = np.arange(0, len(thetas), speed, dtype=int)

    x = L * np.sin(thetas)
    y = -L * np.cos(thetas)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal", adjustable="box")
    pad = 0.2 * L
    ax.set_xlim(-L - pad, L + pad)
    ax.set_ylim(-L - pad, L + pad)
    ax.set_xticks([]); ax.set_yticks([])

    ax.plot([0], [0], "o", markersize=6, color = 'black')
    rod, = ax.plot([], [], lw=2)
    bob, = ax.plot([], [], "o", markersize=10)

    trail_line = None
    if trail and trail > 0:
        trail_line, = ax.plot([], [], lw=1, alpha=0.5)

    # ---- stroboscopic markers (optional) ----
    strobe_step = 2 * divisor
    strobe_x, strobe_y = [], []

    if show_strobe:
        strobe_scatter = ax.scatter([], [], s=strobe_size)
    else:
        strobe_scatter = None

    txt = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va='top',
        ha='left'
    )

    def init():
        rod.set_data([], [])
        bob.set_data([], [])
        if trail_line is not None:
            trail_line.set_data([], [])
        if strobe_scatter is not None:
            strobe_scatter.set_offsets(np.empty((0, 2)))
        txt.set_text("")
        artists = [rod, bob, txt]
        if trail_line is not None:
            artists.append(trail_line)
        if strobe_scatter is not None:
            artists.append(strobe_scatter)
        return tuple(artists)

    def update(k):
        i = frames[k]

        rod.set_data([0, x[i]], [0, y[i]])
        bob.set_data([x[i]], [y[i]])

        if trail_line is not None:
            j0 = max(0, i - trail * speed)
            trail_line.set_data(x[j0:i+1], y[j0:i+1])

        if show_strobe and (i % strobe_step == 0):
            strobe_x.append(x[i])
            strobe_y.append(y[i])
            strobe_scatter.set_offsets(np.column_stack([strobe_x, strobe_y]))

        txt.set_text(f"t = {t_eval[i]:.2f}")

        artists = [rod, bob, txt]
        if trail_line is not None:
            artists.append(trail_line)
        if strobe_scatter is not None:
            artists.append(strobe_scatter)
        return tuple(artists)

    ani = FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, blit=True,
        interval=delay, repeat = False
    )

    fig._ani = ani
    plt.show(block=True)
    return ani

##############################################################
## Section 4.2: Visual animation of the non-autonomous system.
##############################################################    

def animate_pendulum_with_drift(
    theta0, v0, f0, f_final, rate,
    L=1.0,
    fps=60,
    speed=1.0,
    trail=0,
    divisor=200,
    show_strobe=True,         
    strobe_size=18,
    delay = 0,
    large_rate_multiple = 10000
):
    '''
    Animate the pendulum with drifting forcing (non-autonomous system)
    '''

    times, thetas, _, fs = solve_pendulum_with_drift_all_rates(
        rate=rate,
        theta0=theta0,
        v0=v0,
        f0=f0,
        f_final=f_final,
        omega=1.0,
        nu=0.1,
        divisor=divisor,
        settling_time=1000,
        large_rate_multiple=large_rate_multiple,
        large_rate_limit=1
    )

    if (speed < 1):
        print('Speed is less than 1, setting speed = 1')
        speed = 1
    
    frames = np.arange(0, len(thetas), speed, dtype=int)

    x = L * np.sin(thetas)
    y = -L * np.cos(thetas)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal", adjustable="box")
    pad = 0.2 * L
    ax.set_xlim(-L - pad, L + pad)
    ax.set_ylim(-L - pad, L + pad)
    ax.set_xticks([]); ax.set_yticks([])

    ax.plot([0], [0], "o", markersize=6, color = 'black')
    rod, = ax.plot([], [], lw=2)
    bob, = ax.plot([], [], "o", markersize=10)

    trail_line = None
    if trail and trail > 0:
        trail_line, = ax.plot([], [], lw=1, alpha=0.5)

    # ---- stroboscopic markers (optional) ----
    strobe_step = 2 * divisor
    strobe_x, strobe_y = [], []

    if show_strobe:
        strobe_scatter = ax.scatter([], [], s=strobe_size)
    else:
        strobe_scatter = None

    if (rate <= 1):
        txt = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    else:
        txt = ax.text(0.02, 0.9, "", transform=ax.transAxes)

    def init():
        rod.set_data([], [])
        bob.set_data([], [])
        if trail_line is not None:
            trail_line.set_data([], [])
        if strobe_scatter is not None:
            strobe_scatter.set_offsets(np.empty((0, 2)))
        txt.set_text("")
        artists = [rod, bob, txt]
        if trail_line is not None:
            artists.append(trail_line)
        if strobe_scatter is not None:
            artists.append(strobe_scatter)
        return tuple(artists)

    def update(k):
        i = frames[k]

        rod.set_data([0, x[i]], [0, y[i]])
        bob.set_data([x[i]], [y[i]])

        if trail_line is not None:
            j0 = max(0, i - trail * speed)
            trail_line.set_data(x[j0:i+1], y[j0:i+1])

        if show_strobe and (i % strobe_step == 0):
            strobe_x.append(x[i])
            strobe_y.append(y[i])
            strobe_scatter.set_offsets(np.column_stack([strobe_x, strobe_y]))

        txt.set_text(f"t = {times[i]:.2f}; f = {fs[i]:.5f}")
        if (fs[i] < f_final and rate > 1):
            txt.set_text(txt.get_text() + "\n (drifting...Finer dt due to large rate)")

        artists = [rod, bob, txt]
        if trail_line is not None:
            artists.append(trail_line)
        if strobe_scatter is not None:
            artists.append(strobe_scatter)
        return tuple(artists)

    ani = FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, blit=True,
        interval=delay, repeat = False
    )

    fig._ani = ani
    plt.show(block=True)
    return ani
