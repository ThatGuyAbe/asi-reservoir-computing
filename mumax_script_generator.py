"""
mumax_script_generator.py

Writes the mumax3 .mx3 script that drives the ASI with one benchmark signal.

Takes a field table from rc_benchmark_generator.py (two columns, time and field)
and the geometry PNG from asi_geometry_generator.py, and writes one step per
field value. Every step does the same thing:
    relax at +angle, equilibrate, relax at -angle, equilibrate,
    small out-of-plane field pulse, then record the ringdown
and saves the ringdown to its own table file (table_step_XXXX_B_XXXXXXT.txt),
which is what FMR_heatmap.py reads.

The first step also sets the starting magnetization along the field.

run with: python mumax_script_generator.py
"""

import numpy as np

# benchmark input
FIELD_TABLE = "mackey_glass_table.txt"   # from rc_benchmark_generator.py
TASK        = "mackey_glass"             # name used in the header and the output file
REGION      = "sub2"                     # label of the field window
FIELD_MIN   = 0.0197                     # T, same window the table was made for
FIELD_MAX   = 0.0236                     # T

# geometry, from asi_geometry_generator.py
GEOMETRY_PNG  = "ASI_610x260x30nm_4000x4000_row_IIIIIIIII_mis100pct_s3.5_AFTER.png"
GEOMETRY_NAME = "defect"                 # square, rotational, defect ... goes in the file name
ISLAND_NM     = (610, 260, 30)           # length, width, thickness
FILM_NM       = (4000, 4000)             # film size x, y
GRID          = 512                      # cells per side
EDGE_SMOOTH   = 8

# material (permalloy)
MSAT    = "780e3"                        # A/m
AEX     = "13e-12"                       # J/m
MAX_ERR = "1e-5"

# drive and readout
FIELD_ANGLE_DEG = "45.0"                 # field is applied at +angle then -angle
ALPHA_RELAX     = "0.99"                 # damping while relaxing
ALPHA_EQUIL     = "0.05"                 # damping while equilibrating
ALPHA_FMR       = "0.008"                # damping during the ringdown
EQUIL_TIME      = "20e-9"                # s
PULSE_FIELD     = "1e-4"                 # T, out-of-plane pulse
PULSE_TIME      = "20e-12"               # s
FMR_DT          = "30e-12"               # s, ringdown sample spacing
FMR_STEPS       = "250"                  # ringdown samples per step

OUTPUT = f"asi_fmr_{TASK}_{GEOMETRY_NAME}_{REGION}.mx3"


def read_fields(path):
    """field column of a table from rc_benchmark_generator.py"""
    return np.loadtxt(path, comments="#")[:, 1]


def step_block(k, n, B):
    """one field step. k is the step index, n the number of steps."""
    b = f"{B:.15e}"
    plus = f"vector({b}*cos(FieldAngle_rad), {b}*sin(FieldAngle_rad), 0)"
    minus = f"vector({b}*cos(-FieldAngle_rad), {b}*sin(-FieldAngle_rad), 0)"
    pulse = f"vector({b}*cos(-FieldAngle_rad), {b}*sin(-FieldAngle_rad), {PULSE_FIELD})"
    tag = " — INITIALIZATION" if k == 0 else ""
    fn = f"filename_{k:04d}"
    lines = [
        f"// Step {k:04d}  B={b} T{tag}",
        f'print(sprintf("Step {k + 1}/{n}  B=%.6f T", {b}))',
        f"alpha = {ALPHA_RELAX}",
    ]
    if k == 0:
        lines.append("m = uniform(cos(FieldAngle_rad), sin(FieldAngle_rad), 0)")
    lines += [
        f"B_ext = {plus}",
        "relax()",
        "alpha = alpha_equil",
        "run(equil_time)",
        f"B_ext = {minus}",
        f"alpha = {ALPHA_RELAX}",
        "relax()",
        "alpha = alpha_equil",
        "run(equil_time)",
        "alpha = alpha_fmr",
        f"B_ext = {pulse}",
        f"run({PULSE_TIME})",
        f"B_ext = {minus}",
        f'{fn} := sprintf("table_step_{k:04d}_B_{B:+.6f}T.txt")',
        f'fprintln({fn}, "# t (s)\\tmx\\tmy\\tmz")',
        f"for s_{k:04d}:=0; s_{k:04d}<fmr_steps; s_{k:04d}++ {{",
        "    run(fmr_dt)",
        f'    fprintln({fn}, sprintf("%.12e\\t%.12e\\t%.12e\\t%.12e",',
        "        t, m.comp(0).average(), m.comp(1).average(), m.comp(2).average()))",
        "}",
    ]
    return "\n".join(lines) + "\n"


def make_script(fields):
    n = len(fields)
    L, W, T = ISLAND_NM
    fx, fy = FILM_NM
    window = f"{FIELD_MIN:+.4f} to {FIELD_MAX:+.4f} T ({REGION} region)"
    head = f"""///////////////////////// ASI Reservoir Computing — FMR Driver /////////////////////////
//  Task: {TASK} — {REGION}-region run
//  Window: {window}
//  Geometry: {GEOMETRY_PNG}
//  Island size: {L}x{W}x{T} nm | Lattice: {fx}x{fy} nm | Grid {GRID}x{GRID}
//
//  Per step: relax +angle, equilibrate, relax -angle, equilibrate, FMR pulse, ringdown.
//  Output: one table file per step (global mx, my, mz).
/////////////////////////////////////////////////////////////////////////////////////

sizeX := {fx * 1e-9:.9e}
sizeY := {fy * 1e-9:.9e}
sizeZ := {T * 1e-9:.9e}
SetGridsize({GRID}, {GRID}, 1)
SetCellsize(sizeX/{GRID}, sizeY/{GRID}, sizeZ)
EdgeSmooth = {EDGE_SMOOTH}
setgeom(imageShape("{GEOMETRY_PNG}"))

Msat  = {MSAT}
Aex   = {AEX}
MaxErr = {MAX_ERR}

FieldAngle_deg := {FIELD_ANGLE_DEG}
FieldAngle_rad := FieldAngle_deg * pi / 180.0

alpha_equil := {ALPHA_EQUIL}
alpha_fmr   := {ALPHA_FMR}
equil_time  := {EQUIL_TIME}

fmr_dt    := {FMR_DT}
fmr_steps := {FMR_STEPS}

TableAutosave(0)

print("========================================")
print("ASI FMR RC — {TASK} — {REGION} region {window}")
print("Geometry: {GEOMETRY_PNG}  size {L}x{W}x{T}")
print("========================================")
"""
    return head + "".join("\n" + step_block(k, n, B) for k, B in enumerate(fields))


if __name__ == "__main__":
    fields = read_fields(FIELD_TABLE)
    if fields.min() < FIELD_MIN - 1e-9 or fields.max() > FIELD_MAX + 1e-9:
        print("warning: the table goes outside FIELD_MIN to FIELD_MAX, check the window")
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(make_script(fields))
    print(f"Saved: {OUTPUT}  ({len(fields)} steps, {fields.min():.6f} to {fields.max():.6f} T)")
