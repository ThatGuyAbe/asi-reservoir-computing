"""
ASI_NARMA_prediction.py

NARMA-7 and NARMA-10. the input field of each step is scaled to u in 0..0.5 with
the field window in the .mx3 header, and the NARMA target y(t) is computed from u.
a ridge readout of the FMR spectra [Y(t), Y(t-1), ..., Y(t-n)] then predicts y(t)
at the same step, with n = 7 or 10.

after a washout of 100 steps the rest is split in time order: the first 70 % for
development (fit, then the last 20 % of it to pick the ridge alpha), the rest for
test.

Set INPUT_FOLDER to the folder with the four sub region folders, then run:
    python ASI_NARMA_prediction.py
"""
from datetime import datetime
from pathlib import Path, PurePosixPath
import re
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# settings
INPUT_FOLDER = Path("Del Sub regions")  # holds the four sub region folders
NARMA_ORDERS = [7, 10]  # run in turn. the readout uses as many past spectra as the order
DELAY_STEP = 1
WASHOUT = 100
DEVELOPMENT_FRACTION = 0.70
VALIDATION_FRACTION = 0.20  # last part of development
RIDGE_ALPHAS = np.logspace(0, 12, 37)
MSE_WEIGHT = 0.30  # validation score: 0.30 normalized MSE + 0.70 (1 - CC)
FEATURE_Z_CLIP = 8.0
PLOT_TEST_POINTS = 400
SHOW_PLOTS = False  # figures are saved either way
ASSUME_CHRONOLOGICAL_ROWS = False
LABEL_OVERRIDES = {}  # optional {source_stem: display_label}

NPZ_KEYS = {"fields", "frequencies", "power_matrix"}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def find_sub_regions():
    """each sub region folder and the BENCHMARK runs inside it"""
    if not INPUT_FOLDER.is_dir():
        raise FileNotFoundError(f"INPUT_FOLDER not found: {INPUT_FOLDER.resolve()}")
    pattern = re.compile(rf"(?:^|[_ -]){re.escape(BENCHMARK)}(?=[_ -]|$)", re.I)
    regions = []
    for folder in sorted(INPUT_FOLDER.iterdir()):
        if not folder.is_dir() or not re.search(r"sub\s*\d", folder.name, re.I):
            continue
        outs = sorted(p for p in folder.rglob("*.out") if p.is_dir() and "__MACOSX" not in p.parts
                      and pattern.search(p.name[:-4]))
        paths = []
        for out in outs:
            # grid_cache.npz comes from the plotting scripts and repeats the same spectra
            found = sorted(p for p in out.rglob("*.npz") if not p.name.startswith("._")
                           and p.name != "grid_cache.npz")
            if not found:
                warnings.warn(f"{out.name}: no fmr_data.npz, skipped.")
            paths.extend(found)
        print(f"{folder.name}: {len(paths)} {BENCHMARK} runs")
        if len(paths) != 3:
            warnings.warn(f"{folder.name}: expected 3 geometries, found {len(paths)}.")
        if paths:
            regions.append((folder, paths))
    if not regions:
        raise ValueError(f"{INPUT_FOLDER}: no sub region folders with {BENCHMARK} .out folders.")
    return regions


def source_stem(path, fallback):
    for part in reversed(path.parts[:-1]):
        if part.lower().endswith(".out"):
            return part[:-4]
    name = path.stem if path.stem.lower() != "fmr_data" else fallback
    return re.sub(r"\.out$", "", name, flags=re.I)


def describe_source(stem):
    name = re.sub(r"^asi[_-]fmr[_-]", "", stem, flags=re.I)
    task = re.search(r"(?:^|[_ -])(narma[_ -]?(?:10|7)|volterra)(?=[_ -]|$)", name, re.I)
    prefix, tail = (name[:task.start()], name[task.end():]) if task else ("", name)
    if not task:
        leading_prefix = re.match(r"(UK|USA?|DEL)(?=[_ -])", tail, re.I)
        if leading_prefix:
            prefix, tail = leading_prefix[1], tail[leading_prefix.end():]
    if task and not prefix.strip("_ -"):
        trailing_prefix = re.match(r"[_ -]*(UK|USA?|DEL)(?=[_ -]|$)", tail, re.I)
        if trailing_prefix:
            prefix, tail = trailing_prefix[1], tail[trailing_prefix.end():]
    tokens = re.findall(r"[^_ -]+", tail)
    region = [token.capitalize() for token in tokens if re.fullmatch(r"sub\d+", token, re.I)]
    geometry = [token for token in tokens if not re.fullmatch(r"sub\d+", token, re.I)]
    aliases = {"sqr": "Square", "irreg": "Irregular"}
    geometry_label = " ".join(
        aliases.get(t.lower(), t if any(c.isdigit() for c in t) else t.title()) for t in geometry
    )
    prefix = prefix.strip("_ -").replace("_", " ")
    if prefix.lower() in {"uk", "us", "usa", "del"}:
        prefix = prefix.upper()
    label = " ".join(part for part in [prefix, *region, geometry_label] if part) or stem
    return label, geometry_label or stem


def read_spectra(payload, stem, source, driver):
    with np.load(payload, allow_pickle=False) as data:
        if not NPZ_KEYS.issubset(data.files):
            if PurePosixPath(source).name.lower() == "fmr_data.npz":
                raise ValueError(f"{source}: missing {sorted(NPZ_KEYS - set(data.files))}")
            return None
        fields = np.asarray(data["fields"], dtype=float).reshape(-1)
        frequencies = np.asarray(data["frequencies"], dtype=float).reshape(-1)
        spectra = np.asarray(data["power_matrix"], dtype=float)
        steps = np.asarray(data["steps"], dtype=float).reshape(-1) if "steps" in data.files else None
    expected = (len(fields), len(frequencies))
    if spectra.shape != expected and spectra.T.shape == expected:
        spectra = spectra.T
    if spectra.shape != expected or min(expected) == 0:
        raise ValueError(f"{source}: expected spectra of shape {expected}, got {spectra.shape}.")
    if not all(np.isfinite(a).all() for a in [fields, frequencies, spectra]):
        raise ValueError(f"{source}: fields, frequencies and spectra must be finite.")
    if steps is None:
        if not ASSUME_CHRONOLOGICAL_ROWS:
            raise ValueError(f"{source}: no steps array. Confirm row order and enable ASSUME_CHRONOLOGICAL_ROWS.")
        steps = np.arange(len(fields))
    if len(steps) != len(fields) or not np.isfinite(steps).all() or np.any(steps != np.floor(steps)):
        raise ValueError(f"{source}: expected one integer step per spectrum.")
    order = np.argsort(steps)
    steps = steps[order].astype(np.int64)
    if np.any(np.diff(steps) != 1):
        raise ValueError(f"{source}: duplicate or missing steps would invalidate delay alignment.")
    fields, spectra = fields[order], spectra[order]
    freq_order = np.argsort(frequencies)
    frequencies, spectra = frequencies[freq_order], spectra[:, freq_order]
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError(f"{source}: duplicate frequency bins.")

    target_source = "NPZ fields"
    driver_fields = {int(k): float(v) for k, v in re.findall(
        rf"(?mi)^\s*//\s*Step\s+(\d+)\s+B\s*=\s*({NUMBER})\s+T\b", driver
    )}
    if all(int(k) in driver_fields for k in steps):
        precise = np.array([driver_fields[int(k)] for k in steps])
        if np.allclose(precise, fields, atol=5.1e-7, rtol=0):
            fields, target_source = precise, "MX3 fields matched by step"
        else:
            warnings.warn(f"{stem}: MX3 and NPZ fields disagree. Using NPZ fields.")
    window = re.search(rf"(?mi)^\s*//\s*Window:\s*({NUMBER})\s+to\s+({NUMBER})\s*T\b", driver)
    bounds = tuple(map(float, window.groups())) if window else None
    if bounds and (bounds[1] <= bounds[0] or fields.min() < bounds[0] - 5.1e-7
                   or fields.max() > bounds[1] + 5.1e-7):
        bounds = None
    size = re.search(r"(?mi)^\s*//\s*Island size:\s*([^\r\n]+)", driver)
    label, geometry = describe_source(stem)
    return dict(stem=stem, source=source, base_label=label, geometry=geometry,
                fields=fields, spectra=spectra, steps=steps,
                frequencies=frequencies, bounds=bounds, field_source=target_source,
                dimensions=size[1].strip() if size else "")


def load_datasets(paths):
    records = []
    for path in paths:
        if path.is_dir() or path.suffix.lower() == ".npz":
            npzs = sorted(p for p in path.rglob("*") if p.suffix.lower() == ".npz"
                          and "__MACOSX" not in p.parts and not p.name.startswith("._")) if path.is_dir() else [path]
            for p in npzs:
                stem = source_stem(p, path.stem)
                out_dir = next((q for q in p.parents if q.name.lower().endswith(".out")), None)
                script = out_dir.with_suffix(".mx3") if out_dir else p.with_suffix(".mx3")
                driver = script.read_text(errors="replace") if script.is_file() else ""
                record = read_spectra(p, stem, str(p), driver)
                if record is not None:
                    records.append(record)
        else:
            raise ValueError(f"Unsupported input {path}. Use NPZ or an extracted directory.")
    if not records:
        raise ValueError("No FMR spectra found. Check that each .out folder has fmr_data.npz.")
    overrides = {key.casefold(): val for key, val in LABEL_OVERRIDES.items()}
    # custom labels first, then the ones made from the file names
    records.sort(key=lambda r: (r["stem"].casefold() not in overrides, r["source"]))
    used = set()
    for record in records:
        key = record["stem"].casefold()
        label = overrides.get(key, record["base_label"])
        if label in used:
            label = f"{label} ({record['geometry'].lower()} source)"
        base = label
        number = 2
        while label in used:
            label = f"{base} {number}"
            number += 1
        used.add(label)
        record["label"] = label
    return sorted(records, key=lambda r: r["label"])


def mixed_features(spectra, anchors, fit_last_origin):
    fitting = spectra[:fit_last_origin + 1]
    normalized = (spectra - fitting.mean(axis=0)) / (fitting.std(axis=0) + 1e-12)
    if FEATURE_Z_CLIP is not None:
        np.clip(normalized, -FEATURE_Z_CLIP, FEATURE_Z_CLIP, out=normalized)
    return np.concatenate(
        [normalized[anchors - lag * DELAY_STEP] for lag in range(HISTORY_LEN + 1)], axis=1
    )


def ridge_predictions(x_fit, y_fit, x_eval, alphas):
    """ridge with an intercept, every alpha of the grid from one eigensolve"""
    center = x_fit.mean(axis=0)
    x = x_fit - center
    evaluation = x_eval - center
    y_mean = y_fit.mean()
    y = y_fit - y_mean
    if x.shape[1] <= x.shape[0]:
        eigenvalues, vectors = np.linalg.eigh(x.T @ x)
        rhs = vectors.T @ (x.T @ y)
        projected = evaluation @ vectors
    else:
        eigenvalues, vectors = np.linalg.eigh(x @ x.T)
        rhs = vectors.T @ y
        projected = (evaluation @ x.T) @ vectors
    coefficients = rhs[:, None] / (np.maximum(eigenvalues, 0)[:, None] + alphas)
    return y_mean + projected @ coefficients


def scores(actual, predictions):
    actual = np.asarray(actual)
    predicted = np.asarray(predictions).reshape(len(actual), -1)
    centered_actual = actual - actual.mean()
    centered_pred = predicted - predicted.mean(axis=0)
    denom = np.linalg.norm(centered_actual) * np.linalg.norm(centered_pred, axis=0)
    cc = np.divide(centered_actual @ centered_pred, denom, out=np.zeros_like(denom), where=denom > 0)
    mse = np.mean((actual[:, None] - predicted) ** 2, axis=0)
    rmse = np.sqrt(mse)
    return dict(MSE=mse, RMSE=rmse, CC=np.clip(cc, -1, 1),
                NRMSE_std=rmse / actual.std() if actual.std() > 0 else np.full_like(rmse, np.nan),
                NRMSE_range=rmse / np.ptp(actual) if np.ptp(actual) > 0 else np.full_like(rmse, np.nan))


def present(frame):
    try:
        from IPython.display import display
        display(frame)
    except ImportError:
        print(frame.to_string(index=False))


def save_plot(fig, path):
    fig.savefig(path, dpi=175, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)



def narma_target(u, order):
    # y(t+1) = 0.3 y(t) + 0.05 y(t) (sum of the last `order` y) + 1.5 u(t-order+1) u(t) + 0.1,
    # starting from y = 0
    n = order - 1
    y = np.zeros(len(u))
    for t in range(n, len(u)-1):
        y[t+1] = 0.3*y[t] + 0.05*y[t]*y[t-n:t+1].sum() + 1.5*u[t-n]*u[t] + 0.1
        if not np.isfinite(y[t+1]) or abs(y[t+1]) > 1e6:
            raise ValueError(f"{NAME} target diverged at input index {t+1}. Check the input sequence and field range.")
    return y


def imitate(record):
    fields, spectra, steps = record["fields"], record["spectra"], record["steps"]
    if np.ptp(fields) == 0:
        raise ValueError(f"{record['label']}: {NAME} input scaling requires a varying field.")
    if record["bounds"] is None:
        low, high = fields.min(), fields.max()
        scale_source = "Recorded field min/max (reference notebook convention)"
    else:
        low, high = record["bounds"]
        scale_source = "Declared MX3 field window"
    # the clip only catches rounding at the window edges
    u = 0.5 * np.clip((fields-low)/(high-low), 0, 1)
    y = narma_target(u, HISTORY_LEN)

    anchors = np.arange(max(WASHOUT, HISTORY_LEN*DELAY_STEP), len(fields))
    n_dev = int(DEVELOPMENT_FRACTION*len(anchors))
    n_val = int(VALIDATION_FRACTION*n_dev)
    n_train = n_dev-n_val
    if min(n_train, n_val, len(anchors)-n_dev) < 10:
        raise ValueError(f"{record['label']}: too few rows after washout for train/validation/test.")
    target = y[anchors]  # same step as the present spectrum
    alphas = np.asarray(RIDGE_ALPHAS, dtype=float)

    # pick alpha: scale and fit on the training part, score on validation
    x = mixed_features(spectra, anchors[:n_dev], anchors[n_train-1])
    validation = ridge_predictions(x[:n_train], target[:n_train], x[n_train:], alphas)
    val = scores(target[n_train:n_dev], validation)
    normalized_mse = val["MSE"] / (np.var(target[n_train:n_dev]) + 1e-12)
    objective = MSE_WEIGHT*normalized_mse + (1-MSE_WEIGHT)*(1-val["CC"])
    best = int(np.argmin(objective))
    alpha = float(alphas[best])
    del x, validation

    # refit the scaling and the readout on train + validation
    x = mixed_features(spectra, anchors, anchors[n_dev-1])
    predicted = ridge_predictions(x[:n_dev], target[:n_dev], x, np.array([alpha]))[:, 0]
    identity = dict(Dataset=record["label"])
    stats = dict(identity)
    for part, block in [("Fit", slice(0,n_dev)), ("Test", slice(n_dev,None))]:
        stats.update({f"{part}_{key}": float(value[0]) for key,value
                      in scores(target[block], predicted[block]).items()})
    tuning = dict(**identity, Alpha=alpha, Validation_score=float(objective[best]),
                  Validation_MSE=float(val["MSE"][best]), Validation_CC=float(val["CC"][best]),
                  Train=n_train, Validation=n_val, Refit=n_dev, Test=len(anchors)-n_dev,
                  Bins=spectra.shape[1], Features=x.shape[1],
                  Scale_min_T=low, Scale_max_T=high, Input_scale_source=scale_source)
    search = pd.DataFrame({**identity, "Alpha": alphas, "Validation_score": objective,
                           "Validation_NMSE": normalized_mse, "Validation_MSE": val["MSE"],
                           "Validation_CC": val["CC"]})
    predictions = pd.DataFrame({
        **identity, "Step": steps[anchors], "Input": u[anchors], "Target": target,
        "Imitated": predicted, "Segment": np.where(np.arange(len(anchors))<n_dev, "Fit", "Test"),
    })
    return stats, tuning, search, predictions, u, y


def plot_dataset(record, predictions, u, y, stats, directory):
    label, steps = record["label"], record["steps"]
    caption = "\n".join(s for s in (record["stem"], record["dimensions"]) if s)
    directory.mkdir(parents=True, exist_ok=True)

    def finish(fig, name):
        fig.supxlabel(caption, fontsize=8)
        save_plot(fig, directory/name)

    def style(ax, xlabel, ylabel):
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.22)

    fig, ax = plt.subplots(figsize=(12,4.9), layout="constrained")
    power = record["spectra"]
    lo, hi = np.percentile(power, [1, 99.5])
    intensity = np.clip((power-lo)/(hi-lo), 0, 1) if hi>lo else np.zeros_like(power)
    mesh = ax.pcolormesh(steps, record["frequencies"], intensity.T, shading="auto",
                        cmap="viridis", vmin=0, vmax=1)
    ax.set(title=f"{label} · Full FMR spectrum", xlabel="Input step", ylabel="Frequency (GHz)")
    fig.colorbar(mesh, ax=ax, label="Percentile-scaled power")
    finish(fig, "fmr_heatmap.png")

    fig, ax = plt.subplots(figsize=(12,4.2), layout="constrained")
    ax.plot(steps, u, lw=0.9, label="NARMA input u(t)", color="#4776a5")
    ax.plot(steps, y, lw=1.05, label=f"{NAME} target y(t)", color="#27353c")
    start = predictions["Step"].iloc[0]
    ax.axvline(start, color="0.3", ls=":", lw=1.1, label="First usable step")
    ax.set_title(f"{label} · {NAME} input and target")
    style(ax, "Input step", "Value")
    ax.legend(ncol=3, fontsize=8)
    finish(fig, f"{BENCHMARK}_input_and_target.png")

    fitted = predictions[predictions["Segment"] == "Fit"]
    test = predictions[predictions["Segment"] == "Test"]
    zoom = test.head(PLOT_TEST_POINTS)
    title = (f"Test zoom   |   MSE {stats['Test_MSE']:.3e}   |   "
             f"NRMSE/std {stats['Test_NRMSE_std']:.3f}   |   CC {stats['Test_CC']:.4f}")

    def draw_test(ax):
        ax.plot(zoom["Step"], zoom["Target"], color="#27353c", lw=1.3, label="Target")
        ax.plot(zoom["Step"], zoom["Imitated"], color="#c43135", lw=1.15, ls="--", label="Imitation")
        ax.set_title(title)
        style(ax, "Target step", f"{NAME} output")
        ax.legend(fontsize=8)

    fig, (full, detail) = plt.subplots(2, 1, figsize=(12,7.4), layout="constrained")
    full.plot(steps, y, color="0.65", lw=1.0, label=f"{NAME} target")
    full.plot(fitted["Step"], fitted["Imitated"], color="#2166ac", lw=1.0, label="Train + validation fit")
    full.plot(test["Step"], test["Imitated"], color="#c43135", lw=1.0, label="Held-out imitation")
    full.axvline(start, color="0.45", ls=":", lw=0.9, label="First usable step")
    full.axvline(test["Step"].iloc[0], color="0.15", ls="--", lw=0.9, label="First test target")
    full.set_title("Full timeline")
    style(full, "Target step", f"{NAME} output")
    full.legend(ncol=3, fontsize=8)
    draw_test(detail)
    fig.suptitle(f"{label} · {NAME} ridge imitation")
    finish(fig, f"{BENCHMARK}_full_timeline_and_zoom.png")

    fig, ax = plt.subplots(figsize=(12,4.5), layout="constrained")
    draw_test(ax)
    fig.suptitle(label)
    finish(fig, f"{BENCHMARK}_test_only.png")


def comparison_plots(metrics, tuning, directory):
    table = metrics.merge(tuning[["Dataset", "Validation_score"]], on="Dataset", validate="one_to_one")
    columns = {
        "Test_MSE": "Test MSE",
        "Test_NRMSE_std": "Test NRMSE / standard deviation",
        "Test_NRMSE_range": "Test NRMSE / range",
        "Test_CC": "Test Pearson correlation",
        "Validation_score": "Validation selection score",
    }
    for key, title in columns.items():
        fig, ax = plt.subplots(figsize=(8.3, max(2.7, 1.5+0.50*len(table))), layout="constrained")
        ax.barh(table["Dataset"], table[key], color="#257990")
        ax.invert_yaxis()
        ax.set(title=f"{NAME} imitation · {title}", xlabel=title)
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
        save_plot(fig, directory/f"compare_{key.lower()}.png")


def run(paths, directory):
    values = [HISTORY_LEN, DELAY_STEP, WASHOUT, PLOT_TEST_POINTS]
    if (any(not isinstance(v,(int,np.integer)) for v in values)
            or HISTORY_LEN<0 or DELAY_STEP<1 or WASHOUT<0 or PLOT_TEST_POINTS<1):
        raise ValueError("Check integer history, delay spacing, washout and plot length.")
    if not (0<DEVELOPMENT_FRACTION<1 and 0<VALIDATION_FRACTION<1 and 0<=MSE_WEIGHT<=1):
        raise ValueError("Check split fractions and validation weight.")
    alphas = np.asarray(RIDGE_ALPHAS, dtype=float)
    if alphas.ndim!=1 or not len(alphas) or not np.isfinite(alphas).all() or np.any(alphas<=0):
        raise ValueError("RIDGE_ALPHAS must be a nonempty vector of positive finite values.")
    if FEATURE_Z_CLIP is not None and (not np.isfinite(FEATURE_Z_CLIP) or FEATURE_Z_CLIP<=0):
        raise ValueError("FEATURE_Z_CLIP must be positive or None.")

    records = load_datasets(paths)
    directory.mkdir(parents=True)
    metric_rows, settings_rows, searches, traces = [], [], [], []
    for i, record in enumerate(records, 1):
        print(f"{record['label']} · {record['spectra'].shape} · {record['field_source']}")
        metrics, tuning, search, prediction, u, y = imitate(record)
        metric_rows.append(metrics)
        settings_rows.append(tuning)
        searches.append(search)
        traces.append(prediction)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", record["label"]).strip("_")[:100]
        plot_dataset(record, prediction, u, y, metrics, directory/"plots"/f"{i:02d}_{slug}")

    metrics, tuning = pd.DataFrame(metric_rows), pd.DataFrame(settings_rows)
    prediction = pd.concat(traces, ignore_index=True)
    manifest = pd.DataFrame([{k: r[k] for k in ("label","geometry","stem","source",
                                                "field_source","dimensions")} for r in records])
    configuration = pd.DataFrame([dict(History=HISTORY_LEN, Delay_step=DELAY_STEP, Washout=WASHOUT,
        Development_fraction=DEVELOPMENT_FRACTION, Validation_within_development=VALIDATION_FRACTION,
        MSE_weight=MSE_WEIGHT, CC_weight=1-MSE_WEIGHT, Feature_clip=FEATURE_Z_CLIP,
        Ridge_grid=" ".join(map(str,alphas)))])
    for filename, table in [("metrics",metrics),("fit_settings",tuning),("predictions",prediction),
                            ("ridge_validation",pd.concat(searches,ignore_index=True)),
                            ("datasets",manifest),("settings",configuration)]:
        table.to_csv(directory/f"{filename}.csv", index=False)
    print(f"\nHeld-out {NAME} imitation metrics")
    present(metrics[["Dataset","Test_MSE","Test_RMSE","Test_NRMSE_std","Test_NRMSE_range","Test_CC"]].round(6))
    print("\nSelected readouts and chronological split sizes")
    present(tuning[["Dataset","Alpha","Validation_score","Train","Validation","Refit","Test","Features"]].round(5))
    comparison_plots(metrics, tuning, directory)
    bundle = shutil.make_archive(str(directory), "zip", root_dir=directory)
    print(f"\nResults ZIP: {bundle}")
    return metrics, tuning, prediction, directory


results = {}
for order in NARMA_ORDERS:
    BENCHMARK, NAME, HISTORY_LEN = f"narma{order}", f"NARMA{order}", order
    OUTPUT_ROOT = INPUT_FOLDER / f"{BENCHMARK}_imitation_runs"
    regions = find_sub_regions()
    run_folder = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    for region, paths in regions:
        print(f"\n=== {NAME} | {region.name} ===")
        results[NAME, region.name] = run(paths, run_folder / region.name)
    print(f"\nAll {NAME} results: {run_folder.resolve()}")
