"""
Square ASI geometry generator for mumax3.

Spits out a .png bitmap mask (black islands on white) that you load straight
into a mumax3 script as the geometry.

Four ways to add disorder:
  1. row      alternating regular/irregular rows, fixed or random angle
  2. zone      a spatial region (left/right, top/bottom, center/edge, quadrant)
  3. cluster   local patches of disorder around a few random seed islands
  4. custom    you pick the exact islands by (row, col) and set their angles

Global misalignment rotates every island first, then the disorder-mode angles
get added on top.

Island deletion lets you knock out specific islands by (row, col) to study
topological defects. Works with any disorder mode. In row mode it saves both a
BEFORE (full lattice) and an AFTER (with the deletions).

A few things the layout always enforces:
  - first and last rows are vertical islands
  - more horizontal islands per row than vertical
  - no island touches the film edge
"""

import numpy as np
from PIL import Image

# parameters to edit
island_length    = 430      # nm
island_width     = 160      # nm
island_thickness = 25       # nm
edge_separation  = 170      # nm
lattice_spacing  = island_length + edge_separation
film_x           = 3000     # nm
film_y           = 3000     # nm
cellsize         = 4        # nm per pixel
boundary_margin  = 10       # nm
tip_sharpness    = 3.5      # 2.0 is an ellipse, higher is sharper tips

# global misalignment, hits every island
global_misalignment = 0.0   # deg, rotates the whole lattice

# pick one disorder mode: "row", "zone", "cluster", "custom"
disorder_mode = "row"

# islands to delete, works with any mode
deleted_islands = [(3, 1), (4, 2),]

# row mode
row_pattern = [True, True, True, True, True, True, True, True, True]
row_misalign_mode = "random"   # "fixed" or "random"
row_fixed_angle   = 0.0        # deg, added on top of global

# zone mode
zone_type      = "center"
zone_radius    = 0.5
zone_quadrant  = 1

# cluster mode
n_clusters       = 3
cluster_radius   = 800       # nm

# custom mode: (row, col, angle_deg)
custom_islands = [
    (1, 0, 15.0),
    (1, 2, -10.0),
    (3, 1, 12.0),
    (5, 4, -8.0),
    (4, 0, 7.0),
    (6, 3, -14.0),
]

# shared misalignment knobs
misalign_percent = 100.0
misalign_min     = -15.0    # deg
misalign_max     = 15.0     # deg
random_seed      = None


def island_mask(cx, cy, length, width, angle_deg, sharpness, grid_x, grid_y):
    # superellipse centered at (cx,cy), rotated by angle_deg. sharpness controls
    # how pointy the tips are (2 = plain ellipse).
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx = grid_x - cx
    dy = grid_y - cy
    local_x =  cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    a = length / 2.0
    b = width / 2.0
    return (np.abs(local_x) / a) ** sharpness + (np.abs(local_y) / b) ** sharpness <= 1.0


def is_in_zone(cx, cy, film_x, film_y, zone_type, zone_radius, zone_quadrant):
    # is an island at (cx,cy) inside the chosen zone? used by zone mode.
    fx, fy = film_x / 2.0, film_y / 2.0
    if zone_type == "left":
        return cx < fx
    elif zone_type == "right":
        return cx >= fx
    elif zone_type == "top":
        return cy >= fy
    elif zone_type == "bottom":
        return cy < fy
    elif zone_type == "center":
        r = zone_radius * min(film_x, film_y) / 2.0
        return (cx - fx)**2 + (cy - fy)**2 <= r**2
    elif zone_type == "edges":
        r = zone_radius * min(film_x, film_y) / 2.0
        return (cx - fx)**2 + (cy - fy)**2 > r**2
    elif zone_type == "quadrant":
        if zone_quadrant == 1:
            return cx >= fx and cy >= fy
        elif zone_quadrant == 2:
            return cx < fx and cy >= fy
        elif zone_quadrant == 3:
            return cx < fx and cy < fy
        elif zone_quadrant == 4:
            return cx >= fx and cy < fy
    return False


def draw_mask(islands, chosen, delete_indices, apply_deletes,
              global_angle, Ny, Nx, grid_x, grid_y):
    # OR every island into one boolean mask. global angle hits everyone,
    # then the per-island disorder angle from `chosen` gets added.
    mask = np.zeros((Ny, Nx), dtype=bool)
    for idx, (cx, cy, base_angle, ridx) in enumerate(islands):
        if apply_deletes and idx in delete_indices:
            continue
        angle = base_angle + global_angle
        if idx in chosen:
            angle += chosen[idx]
        mask |= island_mask(cx, cy, island_length, island_width, angle,
                            tip_sharpness, grid_x, grid_y)
    return mask


def save_png(mask, filename):
    # mumax wants black islands on white, and the y-axis flipped
    img_array = np.where(mask[::-1], 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img.save(filename)


rng = np.random.default_rng(random_seed)

row_spacing = lattice_spacing / 2.0
m = boundary_margin

# work out how many islands fit per row.
# vertical rows get an even count, horizontal rows an odd count, and we keep
# more horizontals than verticals.
n_vert_per_row = int((film_x - island_width - 2*m) // lattice_spacing) + 1
if n_vert_per_row % 2 != 0:
    n_vert_per_row -= 1

n_horiz_per_row = int((film_x - island_length - 2*m) // lattice_spacing) + 1
if n_horiz_per_row % 2 == 0:
    n_horiz_per_row -= 1

while n_horiz_per_row <= n_vert_per_row:
    n_vert_per_row -= 2

max_extent_y = film_y - island_length - 2*m
max_total_rows = int(max_extent_y // row_spacing) + 1
if max_total_rows % 2 == 0:
    max_total_rows -= 1

n_vert_rows = (max_total_rows + 1) // 2
n_horiz_rows = max_total_rows // 2
total_row_count = n_vert_rows + n_horiz_rows

# film size in pixels (kept even)
Nx = int(np.ceil(film_x / cellsize)) + int(np.ceil(film_x / cellsize)) % 2
Ny = int(np.ceil(film_y / cellsize)) + int(np.ceil(film_y / cellsize)) % 2
total_x = Nx * cellsize
total_y = Ny * cellsize

xs = np.arange(Nx) * cellsize + cellsize / 2.0
ys = np.arange(Ny) * cellsize + cellsize / 2.0
grid_x, grid_y = np.meshgrid(xs, ys)

center_x = total_x / 2.0
center_y = total_y / 2.0
extent_y = (total_row_count - 1) * row_spacing

# build the row pattern (default to alternating if none given)
if row_pattern is None:
    row_pattern = [bool(r % 2) for r in range(total_row_count)]
else:
    while len(row_pattern) < total_row_count:
        row_pattern.append(False)

# lay down all the islands, alternating vertical and horizontal rows.
# each entry is (x, y, base_angle, row_index); row_col_map lets us look an
# island up later by its (row, col).
islands = []
row_col_map = {}
row_index = 0
vert_placed = 0
horiz_placed = 0
is_vert = True

while vert_placed < n_vert_rows or horiz_placed < n_horiz_rows:
    row_y = center_y - extent_y / 2.0 + row_index * row_spacing

    if is_vert and vert_placed < n_vert_rows:
        n_in_row = n_vert_per_row
        x_start = center_x - (n_in_row - 1) * lattice_spacing / 2.0
        for i in range(n_in_row):
            idx = len(islands)
            islands.append((x_start + i * lattice_spacing, row_y, 90.0, row_index))
            row_col_map[(row_index, i)] = idx
        vert_placed += 1
    elif not is_vert and horiz_placed < n_horiz_rows:
        n_in_row = n_horiz_per_row
        x_start = center_x - (n_in_row - 1) * lattice_spacing / 2.0
        for i in range(n_in_row):
            idx = len(islands)
            islands.append((x_start + i * lattice_spacing, row_y, 0.0, row_index))
            row_col_map[(row_index, i)] = idx
        horiz_placed += 1

    is_vert = not is_vert
    row_index += 1

total_islands = len(islands)

# resolve the (row, col) deletion list into island indices
delete_indices = set()
for (row, col) in deleted_islands:
    key = (row, col)
    if key in row_col_map:
        delete_indices.add(row_col_map[key])
    else:
        row_type = "vert" if row % 2 == 0 else "horiz"
        max_col = n_vert_per_row if row % 2 == 0 else n_horiz_per_row
        if row >= total_row_count:
            print(f"  WARNING: ({row},{col}) invalid — only {total_row_count} rows (0-{total_row_count-1})")
        else:
            print(f"  WARNING: ({row},{col}) invalid — row {row} ({row_type}) has {max_col} cols (0-{max_col-1})")

# print the grid so you can read off the (row, col) of every island
print("=== Grid Layout ===")
for r in range(total_row_count):
    row_type = "V" if r % 2 == 0 else "H"
    n_in_row = n_vert_per_row if r % 2 == 0 else n_horiz_per_row
    cols = []
    for c in range(n_in_row):
        idx = row_col_map.get((r, c))
        if idx in delete_indices:
            cols.append(f"col {c} [DEL]")
        else:
            cols.append(f"col {c}")
    print(f"  Row {r} ({row_type}): {n_in_row} islands  [{', '.join(cols)}]")
print()

# pick the per-island disorder angles. these stack on top of global_misalignment.
chosen = {}

if disorder_mode == "row":
    eligible = set()
    for idx, (cx, cy, ang, ridx) in enumerate(islands):
        if ridx < len(row_pattern) and row_pattern[ridx]:
            eligible.add(idx)
    mode_detail = "".join(["I" if p else "R" for p in row_pattern[:total_row_count]])

    n_eligible = len(eligible)
    n_to_misalign = int(np.round(n_eligible * misalign_percent / 100.0))
    if n_to_misalign > 0 and n_eligible > 0:
        eligible_list = sorted(eligible)
        picked = set(rng.choice(eligible_list, size=n_to_misalign, replace=False))
        if row_misalign_mode == "fixed":
            for p in picked:
                chosen[p] = row_fixed_angle
        else:
            angles = rng.uniform(misalign_min, misalign_max, size=n_to_misalign)
            for p, a in zip(sorted(picked), angles):
                chosen[p] = a

elif disorder_mode == "zone":
    eligible = set()
    for idx, (cx, cy, ang, ridx) in enumerate(islands):
        if is_in_zone(cx, cy, film_x, film_y, zone_type, zone_radius, zone_quadrant):
            eligible.add(idx)
    mode_detail = ""
    if zone_type in ("center", "edges"):
        mode_detail = f"r{zone_radius:.1f}"
    elif zone_type == "quadrant":
        mode_detail = f"q{zone_quadrant}"

    n_eligible = len(eligible)
    n_to_misalign = int(np.round(n_eligible * misalign_percent / 100.0))
    if n_to_misalign > 0 and n_eligible > 0:
        eligible_list = sorted(eligible)
        picked = set(rng.choice(eligible_list, size=n_to_misalign, replace=False))
        angles = rng.uniform(misalign_min, misalign_max, size=n_to_misalign)
        for p, a in zip(sorted(picked), angles):
            chosen[p] = a

elif disorder_mode == "cluster":
    eligible = set()
    seed_indices = rng.choice(total_islands, size=min(n_clusters, total_islands), replace=False)
    for si in seed_indices:
        sx, sy = islands[si][0], islands[si][1]
        for idx, (cx, cy, ang, ridx) in enumerate(islands):
            if (cx - sx)**2 + (cy - sy)**2 <= cluster_radius**2:
                eligible.add(idx)

    n_eligible = len(eligible)
    n_to_misalign = int(np.round(n_eligible * misalign_percent / 100.0))
    if n_to_misalign > 0 and n_eligible > 0:
        eligible_list = sorted(eligible)
        picked = set(rng.choice(eligible_list, size=n_to_misalign, replace=False))
        angles = rng.uniform(misalign_min, misalign_max, size=n_to_misalign)
        for p, a in zip(sorted(picked), angles):
            chosen[p] = a

elif disorder_mode == "custom":
    for (row, col, angle) in custom_islands:
        key = (row, col)
        if key in row_col_map:
            idx = row_col_map[key]
            if idx in delete_indices:
                print(f"  WARNING: ({row},{col}) is deleted, skipping misalignment")
            else:
                chosen[idx] = angle
        else:
            row_type = "vert" if row % 2 == 0 else "horiz"
            max_col = n_vert_per_row if row % 2 == 0 else n_horiz_per_row
            print(f"  WARNING: ({row},{col}) invalid — row {row} ({row_type}) has {max_col} cols (0-{max_col-1})")

n_deleted = len(delete_indices)
print(f"Total islands:    {total_islands}")
print(f"Global misalign:  {global_misalignment:+.1f} deg (applied to all)")
print(f"Deleted:          {n_deleted}  {[d for d in deleted_islands if row_col_map.get(d) in delete_indices]}")
print(f"Active islands:   {total_islands - n_deleted}")
print(f"Disorder mode:    {disorder_mode}")
if disorder_mode == "row":
    print(f"Row misalign:     {row_misalign_mode}" + (f" ({row_fixed_angle:+.1f} deg)" if row_misalign_mode == "fixed" else ""))
print(f"Misaligned:       {len(chosen)} islands (additional to global)")
print(f"Horiz/row: {n_horiz_per_row} (odd) | Vert/row: {n_vert_per_row} (even)")
print(f"Rows: {n_vert_rows} vert + {n_horiz_rows} horiz = {total_row_count}")

# assemble a filename that records what was generated
dims = f"{int(island_length)}x{int(island_width)}x{int(island_thickness)}"
film_str = f"{int(film_x)}x{int(film_y)}"
sharp_str = f"s{tip_sharpness:.1f}"
global_str = f"g{global_misalignment:+.0f}deg" if global_misalignment != 0.0 else ""

if disorder_mode == "row":
    mode_detail = "".join(["I" if p else "R" for p in row_pattern[:total_row_count]])
    mode_str = f"row_{mode_detail}"
    if row_misalign_mode == "fixed":
        mis_str = f"fix{row_fixed_angle:+.0f}deg"
    else:
        mis_str = f"mis{int(misalign_percent)}pct"
elif disorder_mode == "zone":
    mode_str = f"zone_{zone_type}"
    if mode_detail:
        mode_str += f"_{mode_detail}"
    mis_str = f"mis{int(misalign_percent)}pct"
elif disorder_mode == "cluster":
    mode_str = f"cluster-n{n_clusters}-r{int(cluster_radius)}"
    mis_str = f"mis{int(misalign_percent)}pct"
elif disorder_mode == "custom":
    mode_str = f"custom-{len(chosen)}islands"
    mis_str = "manual"

base_parts = [f"ASI_{dims}nm", film_str, mode_str, mis_str]
if global_str:
    base_parts.append(global_str)

# write the PNG(s). row mode with deletions saves a BEFORE and an AFTER so you
# can compare the full lattice against the one with islands removed.
if disorder_mode == "row" and n_deleted > 0:
    before_mask = draw_mask(islands, chosen, delete_indices,
                            apply_deletes=False, global_angle=global_misalignment,
                            Ny=Ny, Nx=Nx, grid_x=grid_x, grid_y=grid_y)
    before_name = "_".join(base_parts + [sharp_str, "BEFORE"]) + ".png"
    save_png(before_mask, before_name)
    print(f"\nSaved BEFORE (full lattice + disorder) -> {before_name}")

    after_mask = draw_mask(islands, chosen, delete_indices,
                           apply_deletes=True, global_angle=global_misalignment,
                           Ny=Ny, Nx=Nx, grid_x=grid_x, grid_y=grid_y)
    del_str = f"del{n_deleted}"
    after_name = "_".join(base_parts + [del_str, sharp_str, "AFTER"]) + ".png"
    save_png(after_mask, after_name)
    print(f"Saved AFTER  (disorder + deletions)    -> {after_name}")

    for idx in sorted(delete_indices):
        rc = [k for k, v in row_col_map.items() if v == idx][0]
        cx, cy, base_angle, ridx = islands[idx]
        orientation = "V" if base_angle == 90.0 else "H"
        print(f"  DELETED ({rc[0]},{rc[1]}) {orientation} at ({cx:.0f},{cy:.0f})")

    for idx in sorted(chosen.keys()):
        if idx not in delete_indices:
            rc = [k for k, v in row_col_map.items() if v == idx][0]
            cx, cy, base_angle, ridx = islands[idx]
            orientation = "V" if base_angle == 90.0 else "H"
            total_angle = global_misalignment + chosen[idx]
            print(f"  MISALIGNED ({rc[0]},{rc[1]}) {orientation} at ({cx:.0f},{cy:.0f}) -- global {global_misalignment:+.1f} + local {chosen[idx]:+.1f} = {total_angle:+.1f} deg")

else:
    mask = draw_mask(islands, chosen, delete_indices,
                     apply_deletes=True, global_angle=global_misalignment,
                     Ny=Ny, Nx=Nx, grid_x=grid_x, grid_y=grid_y)

    parts = list(base_parts)
    if n_deleted > 0:
        parts.append(f"del{n_deleted}")
    parts.append(sharp_str)
    output_png = "_".join(parts) + ".png"

    save_png(mask, output_png)
    print(f"\nSaved -> {output_png}  ({Nx}x{Ny} px)")

    for idx in sorted(delete_indices):
        rc = [k for k, v in row_col_map.items() if v == idx][0]
        cx, cy, base_angle, ridx = islands[idx]
        orientation = "V" if base_angle == 90.0 else "H"
        print(f"  DELETED ({rc[0]},{rc[1]}) {orientation} at ({cx:.0f},{cy:.0f})")

    for idx in sorted(chosen.keys()):
        if idx not in delete_indices:
            rc = [k for k, v in row_col_map.items() if v == idx][0]
            cx, cy, base_angle, ridx = islands[idx]
            orientation = "V" if base_angle == 90.0 else "H"
            total_angle = global_misalignment + chosen[idx]
            print(f"  MISALIGNED ({rc[0]},{rc[1]}) {orientation} at ({cx:.0f},{cy:.0f}) -- global {global_misalignment:+.1f} + local {chosen[idx]:+.1f} = {total_angle:+.1f} deg")