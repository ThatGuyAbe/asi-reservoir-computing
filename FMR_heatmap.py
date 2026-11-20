import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# parameters

# mumax3 .out folder with the ringdown table files in it
INPUT_FOLDER = "asi_FMR_UK_Volterra_rotational_irreg_wt_defects_full_sweep.out"  # set this

# where plots go
OUTPUT_FMR_PLOTS = "fmr_individual_plots"
OUTPUT_HEATMAP   = "fmr_heatmap"

# heatmap frequency range (GHz)
FREQ_MIN = 0
FREQ_MAX = 15

# flip to False to skip the per-step FMR plots, runs a lot faster
SAVE_INDIVIDUAL_PLOTS = True

def analyze_fmr_spectrum(table_path):
    """Read one ringdown table and return (freq, total power spectrum)."""
    try:
        rows = []
        with open(table_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace("\\t", "\t").split("\t")
                if len(parts) == 4:
                    rows.append([float(p) for p in parts])
        data = np.array(rows)
        t  = data[:, 0]
        mx = data[:, 1]
        my = data[:, 2]
        mz = data[:, 3]
        dt = t[1] - t[0]

        def calc_power(m_component):
            n = len(m_component)
            x = np.arange(n)
            try:
                # detrend (kill any linear drift) then fade in the start so the
                # FFT doesn't pick up the initial transient as a fake low-freq peak
                m_c = m_component - np.mean(m_component)
                trend = np.polyval(np.polyfit(x, m_component, 1), x)
                m_ac  = m_component - trend
                fade_len = max(1, n // 10)
                fade = np.ones(n)
                fade[:fade_len] = 0.5 * (1 - np.cos(np.pi * np.arange(fade_len) / fade_len))
                m_ac = m_ac * fade
            except Exception:
                m_ac = m_component - np.mean(m_component)
            window     = np.hanning(n)
            m_windowed = m_ac * window
            fft_result  = np.fft.fft(m_windowed)
            frequencies = np.fft.fftfreq(n, dt)
            positive_freq = frequencies[:n//2]
            power = np.abs(fft_result[:n//2])**2
            return positive_freq, power

        freq, power_mx = calc_power(mx)
        _,    power_my = calc_power(my)
        _,    power_mz = calc_power(mz)
        power_total = power_mx + power_my + power_mz   # sum the three components

        return freq, power_total
    except Exception as e:
        print(f"  Error analyzing {table_path}: {e}")
        return None, None

def extract_step_from_filename(filename):
    """Pull the step index out of names like table_step_0220_B_0.000605T.txt"""
    try:
        match = re.search(r'table_step_(\d+)_B_', filename)
        if match:
            return int(match.group(1))
        return None
    except:
        return None

def extract_field_from_filename(filename):
    """
    Pull the field value out of names like:
      table_step_0220_B_0.000605T.txt
      table_step_000_B_-0.165738T.txt
    """
    try:
        match = re.search(r'_B_([+-]?\d+\.\d+)T', filename)
        if match:
            return float(match.group(1))
        return None
    except:
        return None

# main

print("="*70)
print("  FMR DATA ANALYSIS")
print("="*70)

os.makedirs(OUTPUT_FMR_PLOTS, exist_ok=True)
os.makedirs(OUTPUT_HEATMAP,   exist_ok=True)

# grab every table_step_XXXX_B_XXXXXXT.txt in the folder
table_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "table_step_*.txt")))

if len(table_files) == 0:
    print(f"\nNo table files found in '{INPUT_FOLDER}/'")
    print("  Expected files like: table_step_0220_B_0.000605T.txt")
    print("  Check that INPUT_FOLDER points to the right directory.")
    exit(1)

print(f"\nFound {len(table_files)} table files in '{INPUT_FOLDER}'")
print("-"*70)

all_frequencies   = None
all_power_spectra = []
all_fields        = []
all_steps         = []

# work through each file
for i, table_path in enumerate(table_files):
    filename    = os.path.basename(table_path)
    field_value = extract_field_from_filename(filename)
    step_index  = extract_step_from_filename(filename)

    if field_value is None or step_index is None:
        print(f"  Skipping {filename} - could not extract field/step value")
        continue

    print(f"[{i+1}/{len(table_files)}] Step {step_index:04d}, B = {field_value:+.6f} T ...", end=" ")

    freq, power = analyze_fmr_spectrum(table_path)

    if freq is None:
        print("Failed")
        continue

    if all_frequencies is None:
        all_frequencies = freq
        target_length   = len(freq)

    if len(power) != target_length:
        power = np.interp(all_frequencies, freq, power)  # keep all rows the same length

    all_power_spectra.append(power)
    all_fields.append(field_value)
    all_steps.append(step_index)

    peak_idx  = np.argmax(power)
    peak_freq = all_frequencies[peak_idx] / 1e9
    print(f"Peak: {peak_freq:.3f} GHz")

    # one plot per field step
    if SAVE_INDIVIDUAL_PLOTS:
        fig, ax = plt.subplots(figsize=(10, 6))
        freq_ghz = all_frequencies[:-1] / 1e9
        power_pl = power[:-1]

        ax.plot(freq_ghz, power_pl, 'b-', linewidth=1.5)
        ax.set_xlabel('Frequency (GHz)', fontsize=12)
        ax.set_ylabel('Total Power Spectrum (mx+my+mz)', fontsize=12)
        ax.set_title(f'FMR at B = {field_value:+.4f} T  (Peak: {peak_freq:.3f} GHz)',
                     fontsize=13, fontweight='bold')
        ax.set_xlim(0, freq_ghz[-1])
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FMR_PLOTS, f"FMR_B_{field_value:+.6f}T.png"),
                    dpi=200, bbox_inches='tight')
        plt.close()

print(f"\nAnalyzed {len(all_fields)} files successfully")

if len(all_fields) == 0:
    print("No valid data found. Exiting.")
    exit(1)

all_fields          = np.array(all_fields)
all_steps           = np.array(all_steps)
all_frequencies_ghz = all_frequencies / 1e9
power_matrix        = np.array(all_power_spectra)

# frequency mask, shared by both heatmaps
freq_mask  = (all_frequencies_ghz >= FREQ_MIN) & (all_frequencies_ghz <= FREQ_MAX)
freq_plot  = all_frequencies_ghz[freq_mask]

# heatmap 1: FMR dispersion, sorted by field.
# reads like a normal dispersion measurement, modes shifting with field.
# field values come from the input series so the spacing isn't uniform.

print("\nGenerating Heatmap 1: FMR Dispersion (field-sorted)...")

sort_field        = np.argsort(all_fields)
fields_sorted     = all_fields[sort_field]
power_field_sorted = power_matrix[sort_field]

# normalise each field step to its own peak
power_field_norm = power_field_sorted / np.max(power_field_sorted, axis=1, keepdims=True)
power_field_plot = power_field_norm[:, freq_mask]

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(power_field_plot.T,
               aspect='auto', origin='lower',
               extent=[fields_sorted[0], fields_sorted[-1],
                       freq_plot[0], freq_plot[-1]],
               cmap='viridis', vmin=0, vmax=1,
               interpolation='nearest')
ax.set_xlabel('Field (T)',  fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency (GHz)',    fontsize=13, fontweight='bold')
ax.set_title('FMR Dispersion',
             fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Normalized Power', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_HEATMAP, 'FMR_dispersion_field_sorted.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: FMR_dispersion_field_sorted.png")

# heatmap 2: temporal response, sorted by step index.
# this is the order the input actually drives the lattice, which is what
# matters for RC. the modes act on the temporal stream, not a sorted field axis.

print("Generating Heatmap 2: MG Temporal Response (time-ordered)...")

sort_time         = np.argsort(all_steps)
steps_sorted      = all_steps[sort_time]
fields_time_order = all_fields[sort_time]
power_time_sorted = power_matrix[sort_time]

# normalise each step to its own peak
power_time_norm = power_time_sorted / np.max(power_time_sorted, axis=1, keepdims=True)
power_time_plot = power_time_norm[:, freq_mask]

# input waveform
fig_mg, ax_mg = plt.subplots(figsize=(12, 4))
ax_mg.plot(steps_sorted, fields_time_order, 'k-', linewidth=0.5)
ax_mg.set_xlabel('Step Index', fontsize=13, fontweight='bold')
ax_mg.set_ylabel('B (T)', fontsize=13, fontweight='bold')
ax_mg.set_title('Input Waveform', fontsize=14, fontweight='bold')
ax_mg.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_HEATMAP, 'input_waveform.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: input_waveform.png")

# temporal heatmap
fig2, ax2 = plt.subplots(figsize=(12, 8))
im2 = ax2.imshow(power_time_plot.T,
                  aspect='auto', origin='lower',
                  extent=[steps_sorted[0], steps_sorted[-1],
                          freq_plot[0], freq_plot[-1]],
                  cmap='viridis', vmin=0, vmax=1,
                  interpolation='nearest')
ax2.set_xlabel('Step Index', fontsize=13, fontweight='bold')
ax2.set_ylabel('Frequency (GHz)',         fontsize=13, fontweight='bold')
ax2.set_title('Driven FMR',
              fontsize=14, fontweight='bold')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Normalized Power', fontsize=12)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_HEATMAP, 'temporal_response.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: temporal_response.png")

# dump the raw arrays so a later analysis step can pick them up

np.savez(os.path.join(OUTPUT_HEATMAP, 'fmr_data.npz'),
         fields=all_fields,
         steps=all_steps,
         frequencies=freq_plot,
         power_matrix=power_matrix[:, freq_mask])
print("  Saved: fmr_data.npz (includes step indices)")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)