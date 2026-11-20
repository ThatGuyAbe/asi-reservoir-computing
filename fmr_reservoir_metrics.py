"""
fmr_reservoir_metrics.py

Nonlinearity (NL), memory capacity (MC) and complexity (CP) of each ASI device,
from the FMR spectra in fmr_data.npz (one spectrum per input step).

Metrics follow Love et al., arXiv:2108.01512 (eqs. 3-6, sec. II.C, appendix A).
Two readouts:
    no embedding   the present spectrum alone
    mixed          the present spectrum plus the previous HISTORY_LEN spectra
                   (Ti et al., arXiv:2412.04622, sec. III.A)
Each is scored on a train/test split and with no split (fit and score on the
same rows).

Set INPUT_FOLDER to the folder with the sub region folders, then run:
    python fmr_reservoir_metrics.py
needs numpy, pandas, matplotlib
"""

from pathlib import Path, PurePosixPath
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# settings
INPUT_FOLDER = Path("UK Sub regions")           # holds the sub region folders
OUTPUT_DIR = INPUT_FOLDER / "fmr_results"
BENCHMARKS = ["mackey_glass", "narma7", "narma10", "volterra"]
HISTORY_LEN = 10                 # past spectra in the mixed readout
DELAY_STEP = 1
K_NL = 30                        # input history used for NL
K_MC = 30                        # MC sums delays 1 to K_MC
WASHOUT = 100                    # first steps dropped
TRAIN_FRAC = 0.75                # first 75 % train, last 25 % test
VAL_FRAC_WITHIN_TRAIN = 0.20     # ridge alpha is picked inside the training part
CURRENT_MC_SOLVER = "ols"        # no-embedding readout, least squares as in the paper
MIXED_MC_SOLVER = "ridge"        # mixed readout, "ridge" or "ols"
RIDGE_ALPHAS = np.logspace(0, 12, 37)
NMSE_WEIGHT, CC_WEIGHT = 0.30, 0.70    # validation score
FREQUENCY_BAND = None            # None = all bins, or (low, high) in GHz
FREQUENCY_UNIT = "GHz"
ASSUME_ROW_ORDER_IF_NO_STEPS = False   # set True only if the rows are in time order
USE_PRECISE_MX3_INPUT = True     # take the full-precision fields from the .mx3
SHOW_TABLES = True


def dataset_title(name):
    """label from the .out name, e.g. UK Sub1 Square"""
    name = re.sub(r"\.(?:npz|zip|7z)$", "", str(name), flags=re.I)
    name = re.sub(r"\.out$", "", name, flags=re.I)
    raw = [x for x in re.split(r"[_\s.-]+", name) if x]
    lower = [x.lower() for x in raw]
    excluded = {"asi", "fmr", "volterra", "std", "standard"}
    regions = [x for x in lower if re.fullmatch(r"sub\d+", x)]
    boundary = lower.index("volterra") if "volterra" in lower else 0
    prefix_indices = {i for i, x in enumerate(lower)
                      if (i < boundary or x in {"uk", "us", "usa"})
                      and x not in excluded and x not in regions}
    prefix = [x.upper() if len(x) <= 3 else x
              for i, x in enumerate(raw) if i in prefix_indices]
    tokens = [x for i, x in enumerate(lower) if i not in prefix_indices
              and x not in regions and x not in excluded]
    tokens = [{"sqr": "square", "irreg": "irregular"}.get(x, x) for x in tokens]
    geometry = " ".join(tokens)
    return " ".join([*prefix, *(r.capitalize() for r in regions),
                     geometry.title() or "FMR"])


def source_name(path, fallback):
    p = PurePosixPath(str(path).replace("\\", "/"))
    for part in reversed(p.parts):
        if part.lower().endswith(".out"):
            return part
    return p.stem if p.stem.lower() != "fmr_data" else fallback


def find_sub_regions():
    """walk INPUT_FOLDER and sort each sub region's npz files by benchmark"""
    if not INPUT_FOLDER.is_dir():
        raise FileNotFoundError(f"INPUT_FOLDER not found: {INPUT_FOLDER.resolve()}")
    regions = []
    for folder in sorted(INPUT_FOLDER.iterdir()):
        if not folder.is_dir() or folder.name == "__MACOSX" or folder.resolve() == OUTPUT_DIR.resolve():
            continue
        # grid_cache.npz comes from the plotting scripts and has a different band
        npzs = sorted(p for p in folder.rglob("*.npz") if not p.name.startswith("._")
                      and p.name != "grid_cache.npz")
        if not npzs:
            print(f"Skipping {folder.name}: no FMR NPZ found")
            continue
        scripts = {p.stem.lower(): p for p in folder.rglob("*.mx3") if not p.name.startswith("._")}
        found = {}
        for p in npzs:
            name = source_name(p, folder.name).lower()
            hits = [b for b in BENCHMARKS if b in name]
            if len(hits) != 1:
                raise ValueError(f"{p}: name must contain one of {BENCHMARKS}.")
            found.setdefault(hits[0], []).append(p)
        found = {b: found[b] for b in BENCHMARKS if b in found}
        print(f"{folder.name}: " + ", ".join(f"{b} x{len(v)}" for b, v in found.items()))
        if len(found) != len(BENCHMARKS) or any(len(v) != 3 for v in found.values()):
            warnings.warn(f"{folder.name}: expected {len(BENCHMARKS)} benchmarks with 3 geometries each.")
        regions.append((folder, found, scripts))
    if not regions:
        raise ValueError(f"{INPUT_FOLDER}: no sub region folders with FMR NPZ files.")
    return regions


def load_npz(payload, name, source, mx3_text=None):
    with np.load(payload, allow_pickle=False) as d:
        needed = {"fields", "frequencies", "power_matrix"}
        if not needed.issubset(d.files):
            raise ValueError(f"{source}: NPZ must contain {sorted(needed)}; found {d.files}.")
        u = np.asarray(d["fields"], dtype=float).reshape(-1)
        f = np.asarray(d["frequencies"], dtype=float).reshape(-1)
        Y = np.asarray(d["power_matrix"], dtype=float)
        steps = np.asarray(d["steps"]).reshape(-1) if "steps" in d.files else None
    if Y.shape == (len(f), len(u)) and len(f) != len(u):
        Y = Y.T
    if Y.shape != (len(u), len(f)):
        raise ValueError(f"{source}: power_matrix shape {Y.shape} != (samples, bins).")
    if not all(np.isfinite(a).all() for a in (u, f, Y)):
        raise ValueError(f"{source}: non-finite data; repair upstream rather than dropping rows.")
    if steps is None:
        if not ASSUME_ROW_ORDER_IF_NO_STEPS:
            raise ValueError(f"{source}: no steps array. Confirm chronology, then set ASSUME_ROW_ORDER_IF_NO_STEPS=True.")
        warnings.warn(f"{name}: assuming NPZ rows are in chronological order.")
        steps = np.arange(len(u))
    if len(steps) != len(u) or not np.isfinite(steps).all():
        raise ValueError(f"{source}: invalid steps array.")
    if not np.all(steps == np.rint(steps)):
        raise ValueError(f"{source}: steps must be integer input-step indices.")
    order = np.argsort(steps, kind="stable")
    steps, u, Y = steps[order].astype(np.int64), u[order], Y[order]
    if not np.all(np.diff(steps) == 1):
        raise ValueError(f"{source}: duplicate/missing steps; delays require consecutive steps.")
    input_source = "NPZ fields"
    if USE_PRECISE_MX3_INPUT and mx3_text:
        pairs = re.findall(r"//\s*Step\s+(\d+)\s+B\s*=\s*([+\-\d.eE]+)", mx3_text)
        exact = {int(k): float(v) for k, v in pairs}
        if all(int(k) in exact for k in steps):
            candidate = np.array([exact[int(k)] for k in steps])
            if not np.allclose(candidate, u, rtol=0, atol=5.1e-7):
                raise ValueError(f"{name}: MX3 fields do not match NPZ fields within rounding precision.")
            u, input_source = candidate, "MX3 full-precision fields (matched to steps)"
    freq_order = np.argsort(f)
    f, Y = f[freq_order], Y[:, freq_order]
    if np.any(np.diff(f) <= 0):
        raise ValueError(f"{name}: duplicate frequency bins.")
    if FREQUENCY_BAND is not None:
        lo, hi = FREQUENCY_BAND
        keep = (f >= lo) & (f <= hi)
        f, Y = f[keep], Y[:, keep]
    if Y.shape[1] == 0 or np.ptp(u) == 0:
        raise ValueError(f"{name}: empty band or constant input.")
    return dict(name=name, title=dataset_title(name), source=source,
                input_source=input_source, u=u, Y=Y, f=f, steps=steps)


def load_sources(npzs, scripts, region):
    """spectra of every geometry for one benchmark in one sub region"""
    datasets = []
    for p in npzs:
        name = source_name(p, region.name)
        key = re.sub(r"\.out$", "", name, flags=re.I).lower()
        text = scripts[key].read_text(errors="replace") if key in scripts else None
        datasets.append(load_npz(p, name, str(p), text))
    labels = [d["title"] for d in datasets]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate dataset labels detected. Use distinct .out/NPZ names or select each dataset once.")
    return sorted(datasets, key=lambda d: d["title"])


def pearson_columns(actual, predicted):
    """pearson correlation per column. each column is scaled first, since the fmr
    powers are tiny."""
    a, b = np.asarray(actual, float), np.asarray(predicted, float)
    if a.ndim == 1:
        a, b = a[:, None], b[:, None]
    variable_a, variable_b = np.ptp(a, axis=0) > 0, np.ptp(b, axis=0) > 0
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    sa, sb = np.max(np.abs(a), axis=0), np.max(np.abs(b), axis=0)
    a = a / np.where(sa > 0, sa, 1)
    b = b / np.where(sb > 0, sb, 1)
    denom = np.sqrt(np.sum(a*a, axis=0) * np.sum(b*b, axis=0))
    corr = np.full(a.shape[1], np.nan)
    np.divide(np.sum(a*b, axis=0), denom, out=corr,
              where=(denom > 0) & variable_a & variable_b)
    return np.clip(corr, -1, 1)


def fit_linear_svd(X, targets):
    """linear readout by svd. inputs are z-scored on the fit rows and the intercept
    is not penalized. predict_linear gives ols (alpha 0) or ridge."""
    xm, xs = X.mean(axis=0), X.std(axis=0)
    active = np.ptp(X, axis=0) > 0
    xs = np.where(active, xs, 1.0)
    Z = (X - xm) / xs
    Z[:, ~active] = 0.0
    zm = Z.mean(axis=0)
    Z -= zm
    ym, ys = targets.mean(axis=0), targets.std(axis=0)
    variable_target = np.ptp(targets, axis=0) > 0
    ys = np.where(variable_target, ys, 1.0)
    A = (targets - ym) / ys
    A[:, ~variable_target] = 0.0
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    cutoff = (np.finfo(float).eps * max(Z.shape) * s[0]) if s.size else 0
    return dict(xm=xm, xs=xs, zm=zm, ym=ym, ys=ys, s=s, Vt=Vt,
                rhs=U.T @ A, cutoff=cutoff,
                rank=int(np.count_nonzero(s > cutoff)), active=active)


def predict_linear(model, X, alpha=0.0):
    s = model["s"]
    if alpha == 0:
        gain = np.divide(1.0, s, out=np.zeros_like(s), where=s > model["cutoff"])
    else:
        gain = s / (s*s + alpha)
    W = model["Vt"].T @ (gain[:, None] * model["rhs"])
    Z = (X - model["xm"]) / model["xs"] - model["zm"]
    Z[:, ~model["active"]] = 0.0
    return (Z @ W) * model["ys"] + model["ym"]


def validation_score(actual, predicted):
    variance = np.var(actual, axis=0)
    if np.any(np.ptp(actual, axis=0) == 0):
        raise ValueError("Constant validation target; provide a longer random-input sequence.")
    nmse = np.mean((actual - predicted)**2, axis=0) / variance
    cc = pearson_columns(actual, predicted)
    cc = np.nan_to_num(cc, nan=0.0)  # a constant prediction scores as cc = 0
    return float(np.mean(NMSE_WEIGHT * nmse + CC_WEIGHT * (1 - cc)))


def memory_readout(X, U_past, n_train, solver, evaluation="Train/test", alpha=None):
    search = []
    best_alpha = 0.0
    best_score = np.nan
    if evaluation not in {"Train/test", "No split (in-sample)"}:
        raise ValueError("Unknown evaluation variant.")
    no_split = evaluation == "No split (in-sample)"
    if solver == "ridge" and no_split:
        if alpha is None:
            raise ValueError("No-split ridge requires the alpha selected by the train/test variant.")
        best_alpha = float(alpha)
    elif solver == "ridge":
        inner = int(n_train * (1 - VAL_FRAC_WITHIN_TRAIN))
        if min(inner, n_train-inner) < 10:
            raise ValueError("Too few inner train/validation rows.")
        model = fit_linear_svd(X[:inner], U_past[:inner])
        for alpha in RIDGE_ALPHAS:
            prediction = predict_linear(model, X[inner:n_train], float(alpha))
            score = validation_score(U_past[inner:n_train], prediction)
            search.append(dict(alpha=float(alpha), validation_score=score))
        winner = min(search, key=lambda r: r["validation_score"])
        best_alpha, best_score = winner["alpha"], winner["validation_score"]
    elif solver != "ols":
        raise ValueError("MC solvers must be 'ridge' or 'ols'.")
    fit_rows = slice(None) if no_split else slice(0, n_train)
    eval_rows = slice(None) if no_split else slice(n_train, None)
    final = fit_linear_svd(X[fit_rows], U_past[fit_rows])
    predicted = predict_linear(final, X[eval_rows], best_alpha)
    if np.any(np.ptp(U_past[eval_rows], axis=0) == 0):
        raise ValueError("Constant evaluation input target; MC is undefined.")
    r2 = np.nan_to_num(pearson_columns(U_past[eval_rows], predicted), nan=0.0)**2
    return r2, best_alpha, best_score, search, final["rank"]


def complexity(raw_states):
    """effective rank of the last D states (D x D matrix, D = readout coordinates)"""
    D = raw_states.shape[1]
    if len(raw_states) < D:
        return np.nan, np.array([]), "Not enough rows for D x D CP matrix"
    M = raw_states[-D:].copy()
    scale = np.max(np.abs(M))
    if scale == 0:
        return np.nan, np.array([]), "Zero response matrix; effective rank undefined"
    # one common scale factor, so the singular-value weights do not change
    s = np.linalg.svd(M / scale, compute_uv=False)
    p = s[s > 0] / s.sum()
    return float(np.exp(-np.sum(p * np.log(p)))), s, "OK"


def analyze(dataset):
    u, Y, f = dataset["u"], dataset["Y"], dataset["f"]
    max_embedding_lag = HISTORY_LEN * DELAY_STEP
    # a mixed coordinate Y(t-h*d) can depend on u(t-h*d-K_NL), so NL uses the
    # longer input history for both readouts
    nl_horizon = K_NL + max_embedding_lag
    start = WASHOUT + max(nl_horizon, K_MC, max_embedding_lag)
    t = np.arange(start, len(u))
    n_train = int(TRAIN_FRAC * len(t))
    if min(n_train, len(t)-n_train) < 20:
        raise ValueError(f"{dataset['title']}: insufficient samples after washout/history.")
    input_history = np.column_stack([u[t-lag] for lag in range(nl_horizon+1)])
    past_inputs = np.column_stack([u[t-lag] for lag in range(1, K_MC+1)])
    current = Y[t]
    mixed = np.concatenate([Y[t-h*DELAY_STEP] for h in range(HISTORY_LEN+1)], axis=1)
    metric_rows, parameter_rows, searches, profiles, node_rows, spectra = [], [], [], [], [], []
    for embedding, states, solver in [("No embeddings", current, CURRENT_MC_SOLVER),
                                      ("Mixed", mixed, MIXED_MC_SOLVER)]:
        cp, s, cp_status = complexity(states)
        spectra.extend(dict(Dataset=dataset["title"], Embedding=embedding,
                            Index=i+1, Relative_singular_value=float(v/s.sum()))
                       for i, v in enumerate(s))
        split_alpha = None
        for evaluation in ["Train/test", "No split (in-sample)"]:
            no_split = evaluation == "No split (in-sample)"
            fit_rows = slice(None) if no_split else slice(0, n_train)
            eval_rows = slice(None) if no_split else slice(n_train, None)
            print(f"  {dataset['title']} | {embedding} | {evaluation}")
            # NL: ols from the input history to each readout coordinate
            nl_model = fit_linear_svd(input_history[fit_rows], states[fit_rows])
            estimated = predict_linear(nl_model, input_history[eval_rows])
            correlations = pearson_columns(states[eval_rows], estimated)
            valid = (np.ptp(states[eval_rows], axis=0) > 0) & (np.ptp(states[fit_rows], axis=0) > 0)
            node_r2 = np.where(valid, np.nan_to_num(correlations, nan=0.0)**2, np.nan)
            nl = float(1 - np.nanmean(node_r2)) if valid.any() else np.nan
            mc_curve, alpha, score, search, rank = memory_readout(
                states, past_inputs, n_train, solver, evaluation, alpha=split_alpha)
            if not no_split:
                split_alpha = alpha
            identity = dict(Dataset=dataset["title"], Embedding=embedding, Evaluation=evaluation)
            metric_rows.append(dict(**identity, NL=nl, MC=float(mc_curve.sum()), CP=cp))
            parameter_rows.append(dict(**identity,
                N_bins=len(f), D_coordinates=states.shape[1], Samples=len(t),
                Train=len(t) if no_split else n_train, Test=0 if no_split else len(t)-n_train,
                Evaluated=len(t) if no_split else len(t)-n_train,
                Start_step=int(dataset["steps"][start]), NL_input_horizon=nl_horizon,
                MC_max_delay=K_MC, History=HISTORY_LEN if embedding == "Mixed" else 0,
                Delay_step=DELAY_STEP, NL_solver="OLS", MC_solver=solver,
                Ridge_alpha=alpha, Validation_score=score, MC_design_rank=rank,
                Alpha_source=("Fixed zero (OLS)" if solver == "ols" else
                              "Reused from train/test variant" if no_split else "Inner validation"),
                NL_defined_nodes=int(valid.sum()), NL_undefined_nodes=int((~valid).sum()),
                CP_matrix=f"{states.shape[1]} x {states.shape[1]}", CP_status=cp_status,
                Input_source=dataset["input_source"]))
            searches.extend(dict(**identity, **r) for r in search)
            profiles.extend(dict(**identity, Delay=i+1, R2=float(v)) for i, v in enumerate(mc_curve))
            node_rows.extend(dict(**identity, Embedding_lag=(j//len(f))*DELAY_STEP,
                Frequency=float(f[j % len(f)]), Linear_R2=float(v), NL=float(1-v))
                for j, v in enumerate(node_r2))
    return metric_rows, parameter_rows, searches, profiles, node_rows, spectra


def show_table(title, frame, precision=4):
    print("\n" + title)
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.{precision}g}"))


def main():
    for key, value in dict(HISTORY_LEN=HISTORY_LEN, DELAY_STEP=DELAY_STEP,
                           K_NL=K_NL, K_MC=K_MC, WASHOUT=WASHOUT).items():
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    if DELAY_STEP < 1 or K_MC < 1 or not 0 < TRAIN_FRAC < 1:
        raise ValueError("DELAY_STEP/K_MC must be positive; TRAIN_FRAC must be in (0,1).")
    grid = np.asarray(RIDGE_ALPHAS, float)
    if not 0 < VAL_FRAC_WITHIN_TRAIN < 1 or grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all() or np.any(grid < 0):
        raise ValueError("Invalid validation fraction or ridge grid.")
    if any(s not in {"ridge", "ols"} for s in [CURRENT_MC_SOLVER, MIXED_MC_SOLVER]):
        raise ValueError("CURRENT_MC_SOLVER and MIXED_MC_SOLVER must be 'ridge' or 'ols'.")
    if NMSE_WEIGHT < 0 or CC_WEIGHT < 0 or not np.isclose(NMSE_WEIGHT + CC_WEIGHT, 1):
        raise ValueError("Validation weights must be nonnegative and sum to one.")
    results = {}
    for region, found, scripts in find_sub_regions():
        for benchmark, npzs in found.items():
            print(f"\n=== {region.name} | {benchmark} ===")
            datasets = load_sources(npzs, scripts, region)
            results[region.name, benchmark] = analyze_group(
                datasets, OUTPUT_DIR / region.name / benchmark)
    return results


def analyze_group(datasets, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    collections = [[] for _ in range(6)]
    for d in datasets:
        for destination, rows in zip(collections, analyze(d)):
            destination.extend(rows)
    names = ["metrics", "hyperparameters", "ridge_search", "memory_by_delay",
             "nonlinearity_by_node", "cp_singular_values"]
    frames = {name: pd.DataFrame(rows) for name, rows in zip(names, collections)}
    settings = dict(History_len=HISTORY_LEN, Delay_step=DELAY_STEP, K_NL=K_NL, K_MC=K_MC,
        Washout=WASHOUT, Train_fraction=TRAIN_FRAC, Validation_fraction_inside_train=VAL_FRAC_WITHIN_TRAIN,
        Current_MC_solver=CURRENT_MC_SOLVER, Mixed_MC_solver=MIXED_MC_SOLVER,
        Ridge_alphas=[float(a) for a in RIDGE_ALPHAS],
        Validation_NMSE_weight=NMSE_WEIGHT, Validation_CC_weight=CC_WEIGHT,
        Frequency_band=FREQUENCY_BAND, Frequency_unit=FREQUENCY_UNIT,
        CP="Uncentered, unnormalized, final D consecutive raw states; square D x D",
        Evaluation_variants=["Train/test", "No split (in-sample)"],
        NL="1 - mean squared evaluation Pearson correlation; input-history OLS",
        MC="sum squared evaluation Pearson correlation, delays 1..K_MC",
        No_split_rule="Fit and evaluate all aligned rows; reuse alpha from split variant",
        Feature_scaling="Fit-block z-score; all aligned rows for no-split variant",
        Clipping="Off", Input_order="Consecutive step indices; never field magnitude",
        NL_history_rule="K_NL + HISTORY_LEN*DELAY_STEP for both readouts",
        Sources=[dict(source=d["source"], title=d["title"], input_source=d["input_source"])
                 for d in datasets])
    frames["settings"] = pd.DataFrame([(k, str(v)) for k, v in settings.items()],
                                      columns=["Setting", "Value"])
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "run_settings.json").write_text(json.dumps(settings, indent=2))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    cols = min(3, len(datasets))
    dataset_rows = int(np.ceil(len(datasets) / cols))
    rows = 2 * dataset_rows
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.35*rows), squeeze=False,
                             sharex=True, sharey=True, layout="constrained")
    for variant_index, evaluation in enumerate(["Train/test", "No split (in-sample)"]):
        for dataset_index, d in enumerate(datasets):
            ax = axes[variant_index*dataset_rows + dataset_index//cols, dataset_index % cols]
            profile = frames["memory_by_delay"]
            sub = profile.loc[(profile["Dataset"] == d["title"]) & (profile["Evaluation"] == evaluation)]
            for i, (embedding, group) in enumerate(sub.groupby("Embedding", sort=False)):
                solver = CURRENT_MC_SOLVER if embedding == "No embeddings" else MIXED_MC_SOLVER
                label = f"{embedding} / {solver.upper()}"
                ax.plot(group["Delay"], group["R2"], color=["#586778", "#007F9C"][i],
                        linestyle=["--", "-"][i], linewidth=1.8, label=label)
            variant_label = (f"Train/test ({TRAIN_FRAC:.0%}/{1-TRAIN_FRAC:.0%})"
                             if evaluation == "Train/test" else evaluation)
            ax.set(title=f"{d['title']}\n{variant_label}", xlabel="Input delay (steps)",
                   ylabel="Squared correlation", ylim=(-0.02, 1.03),
                   xlim=(0.5, 1.5) if K_MC == 1 else (1, K_MC))
            ax.grid(alpha=0.18)
            ax.legend(frameon=False, fontsize=9)
        for unused in range(len(datasets), dataset_rows*cols):
            axes[variant_index*dataset_rows + unused//cols, unused % cols].set_visible(False)
    fig.suptitle("Linear memory profiles", fontsize=14)
    fig.savefig(output_dir / "memory_profiles.png", dpi=200, bbox_inches="tight")
    if SHOW_TABLES:
        show_table("Metrics (both embeddings x both evaluation variants)", frames["metrics"])
        show_table("Selected readout hyperparameters", frames["hyperparameters"][[
            "Dataset", "Embedding", "Evaluation", "MC_solver", "D_coordinates",
            "Train", "Test", "Ridge_alpha"]], precision=5)
        shared = frames["settings"].query("Setting != 'Sources'").copy()
        shared.loc[shared["Setting"] == "Ridge_alphas", "Value"] = (
            f"{len(RIDGE_ALPHAS)} values, {min(RIDGE_ALPHAS):g} to {max(RIDGE_ALPHAS):g}; full grid in JSON")
        show_table("Shared hyperparameters and other settings", shared)
    plt.close(fig)
    print(f"\nSaved results: {output_dir.resolve()}")
    print("Mixed MC includes stored past responses. Its gain includes explicit external memory.")
    print("No-split metrics use the fitted rows (in-sample); train/test metrics use held-out rows.")
    print("K_NL, K_MC, history and washout are declared settings; verify convergence before scientific claims.")
    if frames["hyperparameters"]["NL_undefined_nodes"].sum():
        warnings.warn("Constant train/test nodes have undefined correlation and are omitted from NL; see hyperparameters.csv.")
    return frames


if __name__ == "__main__":
    results = main()
