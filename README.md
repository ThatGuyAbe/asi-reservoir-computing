# asi-reservoir-computing
Description: Square artificial spin ice as a physical reservoir, with geometry generation, benchmark input signals, and FMR-based readout for MuMax3.

Scripts I use to build square artificial spin ice (ASI) lattices, drive them
with benchmark signals in mumax3, and pull the FMR response back out as
heatmaps. Three Python scripts with a mumax3 step in the middle.

## Pipeline

1. `asi_geometry_generator.py`  build the ASI lattice as a PNG mask
2. *(mumax3)*  load the mask, sweep the field, save the ringdown tables
3. `rc_benchmark_generator.py`  make the field-sweep signals that drive the sim
4. `FMR_heatmap.py`  turn the ringdown tables into FMR heatmaps

Steps 1 and 3 are independent, run them in either order. The mumax3 step needs
both the geometry from step 1 and a field table from step 3.

## Requirements

Python 3 with:

```
pip install numpy matplotlib pillow
```

mumax3 is separate, grab it from mumax.github.io. You run the `.mx3` scripts in
mumax3 yourself. These Python scripts just make the inputs and read the outputs.

## 1. asi_geometry_generator.py

Builds a square ASI lattice and saves it as a black-on-white PNG that mumax3
loads as geometry.

Edit the block at the top (`EDIT PARAMETERS HERE`):
- island size, film size, lattice spacing, cellsize (all nm)
- `disorder_mode`: `"row"`, `"zone"`, `"cluster"`, or `"custom"`
- `global_misalignment`: rotate the whole lattice
- `deleted_islands`: knock out islands by (row, col) for defect studies

Run:

```
python asi_geometry_generator.py
```

You get a PNG named after the settings. It also prints the grid layout so you
can read off the (row, col) of every island before deciding which ones to
misalign or delete. In row mode with deletions you get two PNGs: a BEFORE (full
lattice) and an AFTER (islands removed).

## 2. rc_benchmark_generator.py

Generates the four benchmark signals as field-vs-time tables that drive the
field sweep in mumax3: Mackey-Glass, NARMA10, NARMA7, Volterra.

Edit at the top:
- `FIELD_MIN` / `FIELD_MAX`: the switching window for your lattice. Read these
  off the FMR dispersion heatmap from step 4 (where the modes split or merge).
  For our standard square ASI it lands around -0.10 to +0.05 T.
- `N_POINTS`: number of steps (2000 by default)

Run:

```
python rc_benchmark_generator.py
```

Out: `mackey_glass_table.txt`, `narma10_table.txt`, `narma7_table.txt`,
`volterra_table.txt` (two columns, time and field), plus a preview PNG of all
four signals.

First time through you won't have a heatmap yet, so just use the default field
range, run a sweep, look at the dispersion, then come back and tighten
`FIELD_MIN`/`FIELD_MAX` to the switching region.

## 3. FMR_heatmap.py

Reads the per-step ringdown tables mumax3 dumps during a sweep and turns them
into FMR heatmaps.

Point it at your mumax3 output folder:
- `INPUT_FOLDER`: the `.out` folder full of `table_step_XXXX_B_XXXXXXT.txt` files
- `FREQ_MIN` / `FREQ_MAX`: frequency range for the heatmaps (GHz)
- `SAVE_INDIVIDUAL_PLOTS`: set `False` to skip per-step spectra and run faster

Run:

```
python FMR_heatmap.py
```

Out (in `fmr_heatmap/`):
- `FMR_dispersion_field_sorted.png`: modes vs field, the one you use to set the
  switching window back in step 2
- `temporal_response.png`: spectrum vs step index, the RC-relevant view
- `input_waveform.png`: the driving signal
- `fmr_data.npz`: raw arrays for further analysis

## Typical run

1. Generate a geometry PNG.
2. Generate field tables with the default range.
3. Run the mumax3 sweep with that geometry and one of the tables.
4. Run `FMR_heatmap.py` and look at the dispersion.
5. Set `FIELD_MIN`/`FIELD_MAX` to the switching region, regenerate the tables, rerun.
