"""
ASI_MG_Prediction.py

Mackey-Glass prediction. a ridge readout of the FMR spectra up to step t
(the present spectrum plus the previous HISTORY_LEN) forecasts the input field at
step t+k, for every horizon k in HORIZONS. the field is scaled to 0..1 with the
field window in the .mx3 header, or with the field range before validation if the
header has none.

the steps are split in time order: the first 2/3 for development (fit, then the
last 20 % of it to pick the ridge alpha), the rest for test. a gap of k steps is
left before validation and before test so no target leaks into the fit.

Set INPUT_FOLDER to the folder with the four sub region folders, then run:
    python ASI_MG_Prediction.py
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
INPUT_FOLDER = Path("UK Sub regions")  # holds the four sub region folders
BENCHMARK = "mackey_glass"  # only .out folders with this name are used
OUTPUT_ROOT = INPUT_FOLDER / "mg_prediction_runs"
HORIZONS = [1, 4, 8, 12, 16, 20, 24, 28, 30, 32, 48, 64]
HISTORY_LEN = 50
DELAY_STEP = 1
WASHOUT = 0
DEVELOPMENT_FRACTION = 2 / 3
VALIDATION_FRACTION = 0.20
RIDGE_ALPHAS = np.logspace(0, 12, 37)
MSE_WEIGHT = 0.15
FEATURE_Z_CLIP = 8.0
PLOT_HORIZONS = [8, 30]  # if neither is in HORIZONS, the first horizon is plotted
PLOT_TEST_POINTS = 400
ASSUME_CHRONOLOGICAL_ROWS = False  # set True only for npz files in time order with no steps
LABEL_OVERRIDES = {"asi_fmr_uk_mackey_glass_defect_sub1": "UK Sub1 Square"}

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
    task = re.search(r"(?:^|[_ -])(mackey[_ -]?glass|mg|volterra)(?=[_ -]|$)", name, re.I)
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
    series = "Mackey–Glass" if task and task[1].lower() != "volterra" else "Recorded field"
    return label, geometry_label or stem, series


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
    label, geometry, series = describe_source(stem)
    return dict(stem=stem, source=source, base_label=label, geometry=geometry,
                series=series, fields=fields, spectra=spectra, steps=steps,
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
        raise ValueError("No FMR spectra found. Include fmr_data.npz with fields, frequencies and power_matrix.")
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


def forecast(record, horizon):
    fields, spectra, steps = record["fields"], record["spectra"], record["steps"]
    anchors = np.arange(max(WASHOUT, HISTORY_LEN * DELAY_STEP), len(fields) - horizon)
    dev_end = int(len(anchors) * DEVELOPMENT_FRACTION)
    val_start = dev_end - int(dev_end * VALIDATION_FRACTION)
    # leave a gap of one horizon before validation and before test
    fit_end, refit_end = val_start - horizon, dev_end - horizon
    if min(fit_end, refit_end - val_start, len(anchors) - dev_end) < 10:
        raise ValueError(f"{record['label']}: too few chronological rows for horizon {horizon}. "
                         "Reduce history or the forecast horizon, or use a longer sequence.")
    bounds = record["bounds"]
    if bounds is None:
        observed = fields[:anchors[val_start]]
        bounds = (observed.min(), observed.max())
        scale_source = "Fields observed before first validation origin"
    else:
        scale_source = "Declared MX3 field window"
    low, high = bounds
    if high <= low:
        raise ValueError(f"{record['label']}: target scaling needs a nonconstant input.")
    target_positions = anchors + horizon
    targets = (fields[target_positions] - low) / (high - low)
    fit_rows = slice(0, fit_end)
    val_rows = slice(val_start, refit_end)
    refit_rows = slice(0, refit_end)
    test_rows = slice(dev_end, None)

    x = mixed_features(spectra, anchors[:refit_end], anchors[fit_end - 1])
    val_predictions = ridge_predictions(x[fit_rows], targets[fit_rows], x[val_rows], np.asarray(RIDGE_ALPHAS))
    val = scores(targets[val_rows], val_predictions)
    objective = MSE_WEIGHT * val["MSE"] + (1 - MSE_WEIGHT) * (1 - val["CC"])
    best = int(np.argmin(objective))
    alpha = float(RIDGE_ALPHAS[best])
    del x, val_predictions

    x = mixed_features(spectra, anchors, anchors[refit_end - 1])
    # refit on train + validation, then predict every row with that one model
    predicted = ridge_predictions(x[refit_rows], targets[refit_rows], x, np.array([alpha]))[:, 0]
    del x
    test = {key: float(value[0]) for key, value in scores(targets[test_rows], predicted[test_rows]).items()}
    identity = dict(Dataset=record["label"], Horizon=horizon)
    tuning = dict(**identity, Alpha=alpha, Validation_score=float(objective[best]),
                  Validation_MSE=float(val["MSE"][best]), Validation_CC=float(val["CC"][best]),
                  Train=fit_end, Validation=refit_end-val_start, Refit=refit_end,
                  Test=len(anchors)-dev_end, Gap=horizon, Bins=spectra.shape[1],
                  Features=(HISTORY_LEN+1)*spectra.shape[1], Scale_min_T=low, Scale_max_T=high,
                  Target_scale=scale_source)
    search = pd.DataFrame({**identity, "Alpha": RIDGE_ALPHAS, "Validation_score": objective,
                           "Validation_MSE": val["MSE"], "Validation_CC": val["CC"]})
    predictions = pd.DataFrame({
        **identity, "Origin_step": steps[anchors],
        "Target_step": steps[target_positions], "Actual": targets,
        "Predicted": predicted, "Actual_T": fields[target_positions],
        "Predicted_T": low + predicted * (high-low),
    })
    return dict(**identity, **test), tuning, search, predictions.iloc[test_rows], predictions.iloc[refit_rows]


def present(frame):
    try:
        from IPython.display import display
        display(frame)
    except ImportError:
        print(frame.to_string(index=False))


def save_plot(fig, path):
    fig.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(fig)


def make_plots(datasets, metrics, tuning, predictions, fitted, run_dir, horizons):
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    for i, record in enumerate(datasets):
        group = metrics[metrics["Dataset"] == record["label"]]
        for ax, column, title in zip(axes, ["RMSE", "CC"], ["Test RMSE", "Test Pearson CC"]):
            ax.plot(group["Horizon"], group[column], "o-", color=colors(i % 10),
                    label=record["label"], markersize=4)
            ax.set(xlabel="Forecast horizon (input steps)", ylabel=title)
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Mackey–Glass forecasts by horizon" if all(r["series"] == "Mackey–Glass" for r in datasets)
                 else "Recorded field forecasts by horizon")
    save_plot(fig, run_dir / "forecast_horizons.png")

    shown = [h for h in dict.fromkeys(PLOT_HORIZONS) if h in horizons] or [horizons[0]]
    for i, record in enumerate(datasets, 1):
        label = record["label"]
        slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")[:100]
        for horizon in shown:
            test = predictions[(predictions["Dataset"] == label) &
                               (predictions["Horizon"] == horizon)]
            fit = fitted[(fitted["Dataset"] == label) & (fitted["Horizon"] == horizon)]
            row = metrics[(metrics["Dataset"] == label) & (metrics["Horizon"] == horizon)].iloc[0]
            setting = tuning[(tuning["Dataset"] == label) & (tuning["Horizon"] == horizon)].iloc[0]
            low, high = setting["Scale_min_T"], setting["Scale_max_T"]
            measured = (record["fields"] - low) / (high - low)
            fig, (full, zoom) = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
            full.plot(record["steps"], measured, color="0.65", lw=1.1, label="Measured series")
            full.plot(fit["Target_step"], fit["Predicted"], color="#2166ac", lw=1.1,
                      label="Train + validation fit")
            full.plot(test["Target_step"], test["Predicted"], color="#c43135", lw=1.1,
                      label="Held-out forecast")
            full.axvline(test["Target_step"].iloc[0], color="0.2", ls="--", lw=0.9,
                         label="First test target")
            full.set(title="Full timeline", xlabel="Target step", ylabel="Scaled field")
            full.legend(loc="upper right", fontsize=8, ncol=2)
            subset = test.head(PLOT_TEST_POINTS)
            zoom.plot(subset["Target_step"], subset["Actual"], color="#293947", label="Measured field")
            zoom.plot(subset["Target_step"], subset["Predicted"], color="#c43135", ls="--", label="Forecast")
            zoom.set(title=f"Test zoom   |   MSE {row.MSE:.3e}   |   NRMSE/std {row.NRMSE_std:.3f}   |   CC {row.CC:.4f}",
                     xlabel="Target step", ylabel="Scaled field")
            zoom.legend(loc="upper right", fontsize=8)
            for ax in (full, zoom):
                ax.grid(alpha=0.2)
            fig.suptitle(f"{label}  ·  {record['series']} prediction  ·  k={horizon}")
            fig.supxlabel(record["stem"], fontsize=8)
            save_plot(fig, run_dir / f"{i:02d}_{slug}_k{horizon}_forecast_timeline.png")


def run(paths, run_dir):
    if not HORIZONS or any(not isinstance(k, (int, np.integer)) or k <= 0 for k in HORIZONS):
        raise ValueError("HORIZONS must contain positive integers.")
    integers = [HISTORY_LEN, DELAY_STEP, WASHOUT, PLOT_TEST_POINTS]
    if (any(not isinstance(v, (int, np.integer)) for v in integers)
            or HISTORY_LEN < 0 or DELAY_STEP < 1 or WASHOUT < 0 or PLOT_TEST_POINTS < 1):
        raise ValueError("Check history, delay spacing, washout and plot length.")
    if FEATURE_Z_CLIP is not None and (not np.isfinite(FEATURE_Z_CLIP) or FEATURE_Z_CLIP <= 0):
        raise ValueError("FEATURE_Z_CLIP must be positive and finite, or None.")
    if not (0 < DEVELOPMENT_FRACTION < 1 and 0 < VALIDATION_FRACTION < 1 and 0 <= MSE_WEIGHT <= 1):
        raise ValueError("Check split fractions and validation weight.")
    alphas = np.asarray(RIDGE_ALPHAS)
    if alphas.ndim != 1 or not alphas.size or not np.isfinite(alphas).all() or (alphas <= 0).any():
        raise ValueError("RIDGE_ALPHAS must be a nonempty vector of positive finite values.")
    horizons = sorted(set(HORIZONS))
    datasets = load_datasets(paths)
    run_dir.mkdir(parents=True)
    metrics, tuning, searches, predictions, fitted = [], [], [], [], []
    for record in datasets:
        rows, bins = record["spectra"].shape
        print(f"{record['label']} · {rows:,} steps × {bins:,} bins · {record['field_source']}")
        if record["series"] != "Mackey–Glass":
            print("  Forecasting the recorded field sequence for this dataset.")
        for k in horizons:
            metric, parameters, search, prediction, fit = forecast(record, k)
            metrics.append(metric)
            tuning.append(parameters)
            searches.append(search)
            predictions.append(prediction)
            fitted.append(fit)
        print(f"  Completed {len(horizons)} horizons.")
    metrics, tuning = pd.DataFrame(metrics), pd.DataFrame(tuning)
    predictions = pd.concat(predictions, ignore_index=True)
    fitted = pd.concat(fitted, ignore_index=True)
    pd.DataFrame([{k: r[k] for k in ["label", "geometry", "stem", "source", "series",
                                     "field_source", "dimensions"]} for r in datasets]).to_csv(
        run_dir / "datasets.csv", index=False
    )
    pd.DataFrame([dict(History=HISTORY_LEN, Delay_step=DELAY_STEP, Washout=WASHOUT,
                       Horizons=" ".join(map(str, horizons)), Development=DEVELOPMENT_FRACTION,
                       Validation_within_development=VALIDATION_FRACTION, MSE_weight=MSE_WEIGHT,
                       Feature_z_clip=FEATURE_Z_CLIP)]).to_csv(run_dir / "settings.csv", index=False)
    metrics.to_csv(run_dir / "test_metrics.csv", index=False)
    tuning.to_csv(run_dir / "fit_settings.csv", index=False)
    pd.concat(searches, ignore_index=True).to_csv(run_dir / "ridge_validation.csv", index=False)
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    fitted.to_csv(run_dir / "fitted_predictions.csv", index=False)
    print("\nHeld-out forecast metrics. RMSE is on the scaled field.")
    present(metrics[["Dataset", "Horizon", "RMSE", "CC", "NRMSE_std"]].round(5))
    print("\nSelected readouts and split sizes")
    present(tuning[["Dataset", "Horizon", "Alpha", "Train", "Validation", "Refit", "Test"]].round(4))
    make_plots(datasets, metrics, tuning, predictions, fitted, run_dir, horizons)
    bundle = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    print(f"\nResults ZIP: {bundle}")
    return metrics, tuning, predictions, run_dir


regions = find_sub_regions()
run_folder = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
results = {}
for region, paths in regions:
    print(f"\n=== {region.name} ===")
    results[region.name] = run(paths, run_folder / region.name)
print(f"\nAll results: {run_folder.resolve()}")