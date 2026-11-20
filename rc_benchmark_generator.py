"""
rc_benchmark_generator.py

Builds the input field tables that get fed into mumax3 for the RC benchmarks.

Four signals come out of this:
    mackey_glass   chaotic delay diff-eq, the standard RC memory task
    narma10        order-10 NARMA
    narma7         shorter order-7 version
    volterra       2nd-order (quadratic) Volterra system

Everything gets normalised into the field window where the ASI actually
switches. Grab FIELD_MIN / FIELD_MAX off the FMR dispersion heatmap: the
switching region is wherever the modes split or merge. For our standard
square ASI that's roughly -0.10 to +0.05 T.

run with: python rc_benchmark_generator.py
"""

import numpy as np
import matplotlib.pyplot as plt

# config: point these at your switching region

# Field window for the signals, read off the FMR dispersion heatmap.
# The switching region is where the modes split/merge.
FIELD_MIN = -0.12   # T, bottom of the switching window
FIELD_MAX = 0.02    # T, top of the switching window

N_POINTS = 2000     # steps for the mumax sweep
SEED = 42

# signal generators

def mackey_glass(n_points=10000, tau=17, beta=0.2, gamma=0.1, n=10, dt=0.1):
    history_len = int(tau / dt) + 1
    x = np.zeros(n_points + history_len)
    x[0:history_len] = 0.5 + 0.1 * np.random.rand(history_len)
    for i in range(history_len, n_points + history_len):
        x_tau = x[i - int(tau / dt)]
        k1 = beta * x_tau / (1 + x_tau**n) - gamma * x[i-1]
        k2 = beta * x_tau / (1 + x_tau**n) - gamma * (x[i-1] + 0.5*dt*k1)
        k3 = beta * x_tau / (1 + x_tau**n) - gamma * (x[i-1] + 0.5*dt*k2)
        k4 = beta * x_tau / (1 + x_tau**n) - gamma * (x[i-1] + dt*k3)
        x[i] = x[i-1] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return x[history_len:]


def narma(n_points=10000, order=10, alpha=0.3, beta=0.05, gamma=1.5, delta=0.1):
    """
    General NARMA-n. order=10 gives classic NARMA10, order=7 gives NARMA7.

    y(t) = tanh( alpha*y(t-1) + beta*y(t-1)*sum_{i=1..order} y(t-i)
                 + gamma*u(t-(order-1))*u(t) + delta )

    Input u ~ Uniform(0, 0.5). The tanh is the bounded version: the bare
    recursion blows up for some input draws, which trashes the normalised
    table. order=10 reproduces the original narma10.
    """
    u = np.random.uniform(0, 0.5, n_points + order)
    y = np.zeros(n_points + order)
    y[0:order] = 0.1
    for t in range(order, n_points + order):
        y_sum = np.sum(y[t-order:t])              # last `order` outputs
        y[t] = np.tanh(alpha * y[t-1] +
                       beta  * y[t-1] * y_sum +
                       gamma * u[t-(order-1)] * u[t] +
                       delta)
    return y[order:]


def volterra(n_points=10000):
    """
    Truncated 2nd-order (quadratic) Volterra system, memory M=2:

        y(t) = sum_i h1[i] u(t-i) + sum_{i<=j} h2[i,j] u(t-i) u(t-j)

    Standard discrete 2nd-order Volterra series (Koh & Powers form). Volterra
    is a whole class of systems, so the kernels below are just one fixed pick:
    a dominant linear part with a weaker quadratic correction (fading memory).
    Same Uniform(0, 0.5) input as NARMA so the two stay comparable.
    """
    u = np.random.uniform(0, 0.5, n_points + 2)

    # linear kernel h1[i], i = 0,1,2
    h1 = np.array([1.0, 0.5, -0.2])
    # quadratic kernel h2[i,j], i <= j (symmetric)
    h2 = {(0, 0): 0.3, (0, 1): -0.4, (0, 2): 0.1,
          (1, 1): 0.2, (1, 2): -0.1,
          (2, 2): 0.05}

    y = np.zeros(n_points)
    for k in range(n_points):
        t = k + 2                                 # current index into u
        lin = h1[0]*u[t] + h1[1]*u[t-1] + h1[2]*u[t-2]
        quad = sum(c * u[t-i] * u[t-j] for (i, j), c in h2.items())
        y[k] = lin + quad
    return y


def normalize(signal, lo, hi):
    """Squash signal into [lo, hi]."""
    s = signal - signal.min()
    s = s / s.max()  # now [0, 1]
    return s * (hi - lo) + lo


# generate the signals

# Make more points than we need and keep the first N_POINTS. Reseed before
# each one so the four datasets are independent but still reproducible.
# mackey-glass moves slowly at dt = 0.1, so the first N_POINTS samples only cover a
# few oscillations. integrate longer and keep every MG_DECIM-th sample.
MG_DECIM = 4
np.random.seed(SEED)
mg_raw = mackey_glass(n_points=N_POINTS * MG_DECIM + MG_DECIM)[::MG_DECIM]

np.random.seed(SEED)
narma10_raw = narma(n_points=max(N_POINTS * 5, 10000), order=10)

np.random.seed(SEED + 1)
narma7_raw = narma(n_points=max(N_POINTS * 5, 10000), order=7)

np.random.seed(SEED + 2)
volterra_raw = volterra(n_points=max(N_POINTS * 5, 10000))

# normalise into the target field window
mg_field       = normalize(mg_raw[:N_POINTS],       FIELD_MIN, FIELD_MAX)
narma10_field  = normalize(narma10_raw[:N_POINTS],  FIELD_MIN, FIELD_MAX)
narma7_field   = normalize(narma7_raw[:N_POINTS],   FIELD_MIN, FIELD_MAX)
volterra_field = normalize(volterra_raw[:N_POINTS], FIELD_MIN, FIELD_MAX)

dt_mumax = 5e-12
t_s  = np.arange(N_POINTS) * dt_mumax
t_ns = t_s * 1e9

print(f"Field range: [{FIELD_MIN}, {FIELD_MAX}] T")
print(f"Steps: {N_POINTS}")
print(f"MG field:       [{mg_field.min():.4f}, {mg_field.max():.4f}] T")
print(f"NARMA-10 field: [{narma10_field.min():.4f}, {narma10_field.max():.4f}] T")
print(f"NARMA-7 field:  [{narma7_field.min():.4f}, {narma7_field.max():.4f}] T")
print(f"Volterra field: [{volterra_field.min():.4f}, {volterra_field.max():.4f}] T")

# write the tables

for sig, name in [(mg_field,       "mackey_glass"),
                  (narma10_field,  "narma10"),
                  (narma7_field,   "narma7"),
                  (volterra_field, "volterra")]:
    fname = f"{name}_table.txt"
    with open(fname, "w") as f:
        f.write("# time_s              field_T\n")
        for t, B in zip(t_s, sig):
            f.write(f"{t:.15e} {B:.15e}\n")
    print(f"Saved: {fname}")

# preview figure

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

for ax, sig, color, title in [
    (axes[0], mg_field,       '#1a6fbf', f'Mackey-Glass [{FIELD_MIN}, {FIELD_MAX}] T'),
    (axes[1], narma10_field,  '#c0392b', f'NARMA-10 [{FIELD_MIN}, {FIELD_MAX}] T'),
    (axes[2], narma7_field,   '#e67e22', f'NARMA-7 [{FIELD_MIN}, {FIELD_MAX}] T'),
    (axes[3], volterra_field, '#27ae60', f'Volterra [{FIELD_MIN}, {FIELD_MAX}] T'),
]:
    ax.plot(np.arange(N_POINTS), sig, color=color, linewidth=0.8)
    ax.set_ylabel("Field (T)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(FIELD_MIN - 0.01, FIELD_MAX + 0.01)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.grid(True, alpha=0.3)

axes[3].set_xlabel("Step", fontsize=11)
plt.tight_layout()
plt.savefig("rc_benchmark_signals.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: rc_benchmark_signals.png")

print(f"\nDone! Update FIELD_MIN and FIELD_MAX based on your FMR dispersion heatmap.")
print(f"Then re-run this script and pass the new tables to mumax_script_generator.py.")
