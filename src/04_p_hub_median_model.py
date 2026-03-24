import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum
from collections import defaultdict
from math import inf
import time
import matplotlib; matplotlib.use("Agg")  

# CONFIG: file paths & parameters 
OD_CSV = 'od_distances_test.csv'                        # EITHER has i/j (or from/to) OR lat/lon columns
NODE_HUB_CSV = 'node_to_hub_distance_test.csv'          # long form: NodeID, HubID, Distance_km
HUB_HUB_CSV = 'hub_to_hub_distance_test.csv'            # long form: HubID_From, HubID_To, Distance_km
NODES_CATALOG_CSV = 'nodes_catalog_test.csv'            # NodeID, Latitude, Longitude (used if OD has lat/lon)

# Testing knobs 
TEST_MAX_OD_PAIRS = None   # solve with only first 20 OD pairs (set None to disable)
TEST_MAX_HUBS = None       # limit candidate hubs to 10 (set None to disable)

p = 10                      # number of hubs to open (must be <= number of candidate hubs)
alpha_km = 0.2             # multiplier on inter-hub distances (km)

# Preprocessing / formulation toggles (Hamacher path-based properties)
ALLOW_LOOPS = False          # if True, allow single-hub routes (i==j). If False, loops won't be created in A
USE_UNDIRECTED = True        # if True, for each unordered {i,j} keep only the cheaper direction per k
APPLY_DOMINANCE = True       # if True, drop two-hub edges dominated by single-hub (Property 1)
ASSUME_SYMMETRIC = True      # if True and ALLOW_LOOPS, if o(k)==d(k) keep only loops (Property 2)

def pick_col(cols, *names):
    cl = {c.lower(): c for c in cols}
    for nm in names:
        if nm.lower() in cl: 
            return cl[nm.lower()]
    return None

# Load OD pairs (unit demand per row, aggregated)
od = pd.read_csv(OD_CSV)

# Try ID-based columns first
col_i = pick_col(od.columns, 'i','from','origin','o')
col_j = pick_col(od.columns, 'j','to','destination','d')

if col_i and col_j:
    od = od[[col_i, col_j]].copy()
    od[col_i] = od[col_i].astype(int)
    od[col_j] = od[col_j].astype(int)
    od = od[od[col_i] != od[col_j]]
else:
    # Try coordinate-based columns and map to NodeIDs via nodes_catalog.csv
    lat_o = pick_col(od.columns, 'start_lat','from_lat','origin_lat','o_lat','from_latitude')
    lon_o = pick_col(od.columns, 'start_lon','from_lon','origin_lon','o_lon','from_longitude')
    lat_d = pick_col(od.columns, 'destination_lat','to_lat','dest_lat','d_lat','to_latitude')
    lon_d = pick_col(od.columns, 'destination_lon','to_lon','dest_lon','d_lon','to_longitude')
    if not all([lat_o, lon_o, lat_d, lon_d]):
        raise ValueError("OD_CSV must have either (i,j)/(from,to) columns OR lat/lon pairs: Start/From_* and Destination/To_*")

    # Load nodes catalog and round coordinates for robust join
    nodes = pd.read_csv(NODES_CATALOG_CSV)
    n_lat = pick_col(nodes.columns, 'latitude','lat')
    n_lon = pick_col(nodes.columns, 'longitude','lon')
    n_id  = pick_col(nodes.columns, 'nodeid','node','id')
    if not (n_id and n_lat and n_lon):
        raise ValueError("nodes_catalog.csv must have columns: NodeID, Latitude, Longitude")

    nodes = nodes[[n_id, n_lat, n_lon]].copy()
    nodes['_lat_r'] = nodes[n_lat].astype(float).round(6)
    nodes['_lon_r'] = nodes[n_lon].astype(float).round(6)

    tmp = od[[lat_o, lon_o, lat_d, lon_d]].copy()
    tmp['_o_lat_r'] = tmp[lat_o].astype(float).round(6)
    tmp['_o_lon_r'] = tmp[lon_o].astype(float).round(6)
    tmp['_d_lat_r'] = tmp[lat_d].astype(float).round(6)
    tmp['_d_lon_r'] = tmp[lon_d].astype(float).round(6)

    # Map origin coords -> NodeID
    m_o = tmp.merge(nodes, left_on=['_o_lat_r','_o_lon_r'], right_on=['_lat_r','_lon_r'], how='left')
    if m_o[n_id].isna().any():
        missing = int(m_o[n_id].isna().sum())
        raise ValueError(f"Could not map {missing} origin coordinates to NodeID using {NODES_CATALOG_CSV}. Check rounding or catalog.")
    # Keep only needed columns; drop any merge helper columns if present
    m_o = m_o.drop(columns=['_lat_r','_lon_r', n_lat, n_lon], errors='ignore').rename(columns={n_id: 'i'})

    # Map destination coords -> NodeID
    m_d = m_o.merge(nodes, left_on=['_d_lat_r','_d_lon_r'], right_on=['_lat_r','_lon_r'], how='left')
    if m_d[n_id].isna().any():
        missing = int(m_d[n_id].isna().sum())
        raise ValueError(f"Could not map {missing} destination coordinates to NodeID using {NODES_CATALOG_CSV}. Check rounding or catalog.")
    # Drop helper columns and rename NodeID to 'j'
    m_d = m_d.drop(columns=['_lat_r','_lon_r', n_lat, n_lon], errors='ignore').rename(columns={n_id: 'j'})

    od = m_d[['i','j']].astype(int)
    od = od[od['i'] != od['j']]
    col_i, col_j = 'i', 'j'

# Aggregate unit demand per unique (i,j)
ow = (
    od.groupby([col_i, col_j], as_index=False)
      .size()
      .rename(columns={'size': 'Demand'})
)

# Build OD set and demand dict
OD_pairs = [(int(r[col_i]), int(r[col_j])) for _, r in ow.iterrows()]
w = {(int(r[col_i]), int(r[col_j])): int(r['Demand']) for _, r in ow.iterrows()}
O_nodes = sorted(set([i for i,_ in OD_pairs]) | set([j for _,j in OD_pairs]))

print(f"Loaded {len(OD_pairs)} unique OD pairs; total unit rows = {ow['Demand'].sum()}.")

# Build Hamacher-style commodity index K 
K_ids = list(range(len(OD_pairs)))                 # k = 0..|OD|-1
ok = {k: OD_pairs[k][0] for k in K_ids}            # origin node of commodity k
dk = {k: OD_pairs[k][1] for k in K_ids}            # destination node of commodity k
wk = {k: w[OD_pairs[k]] for k in K_ids}            # demand of commodity k

# Optional: restrict to a small number of OD pairs for testing 
if TEST_MAX_OD_PAIRS is not None:
    OD_pairs = OD_pairs[:TEST_MAX_OD_PAIRS]
    w = {k: w[k] for k in OD_pairs}
    O_nodes = sorted(set([i for i,_ in OD_pairs]) | set([j for _,j in OD_pairs]))
    # rebuild K index
    K_ids = list(range(len(OD_pairs)))
    ok = {k: OD_pairs[k][0] for k in K_ids}
    dk = {k: OD_pairs[k][1] for k in K_ids}
    wk = {k: w[OD_pairs[k]] for k in K_ids}
    print(f"[TEST] Restricted to {len(OD_pairs)} OD pairs; nodes involved: {len(O_nodes)}")

# Load Node→Hub distances (km) 
nh = pd.read_csv(NODE_HUB_CSV)
col_node = pick_col(nh.columns, 'nodeid','node','i','from')
col_hub  = pick_col(nh.columns, 'hubid','hub','k')
col_nd   = pick_col(nh.columns, 'distance_km','distance','dist','km')
if not (col_node and col_hub and col_nd):
    raise ValueError("NODE_HUB_CSV must have columns like NodeID, HubID, Distance_km")

nh = nh[[col_node, col_hub, col_nd]].copy()
nh[col_node] = nh[col_node].astype(int)
nh[col_hub]  = nh[col_hub].astype(int)


# Optional: restrict candidate hubs for testing 
all_hubs = sorted(nh[col_hub].unique().tolist())
if TEST_MAX_HUBS is not None:
    selected_hubs = all_hubs[:TEST_MAX_HUBS]
else:
    selected_hubs = all_hubs

nh = nh[nh[col_hub].isin(selected_hubs)]

# Cost dictionaries
c_ik = {(int(r[col_node]), int(r[col_hub])): float(r[col_nd]) for _, r in nh.iterrows()}
H_hubs = sorted(nh[col_hub].unique().tolist())
print(f"Hubs in NODE_HUB_CSV: {len(H_hubs)} unique IDs")
# Quick sanity on coords if stations.csv present
try:
    _cat = pd.read_csv('stations.csv')
    _latc = [c for c in _cat.columns if c.lower() in ('latitude','lat')]
    _lonc = [c for c in _cat.columns if c.lower() in ('longitude','lon')]
    _idc  = [c for c in _cat.columns if c.lower() in ('hubid','hub_id','hub_index','index','id')]
    if _latc and _lonc:
        latmin, latmax = _cat[_latc[0]].min(), _cat[_latc[0]].max()
        lonmin, lonmax = _cat[_lonc[0]].min(), _cat[_lonc[0]].max()
        print(f"stations.csv lat range=({latmin:.2f},{latmax:.2f}), lon range=({lonmin:.2f},{lonmax:.2f})")
        if (latmin < -10 or latmax > 60) or (lonmin < 60 or lonmax > 110):
            print('[Warning] stations.csv coordinates look out-of-range for India. Check if lat/lon swapped or in DMS.')
    if _idc:
        missing = sorted(set(H_hubs) - set(_cat[_idc[0]].astype(int).unique()))
        if missing:
            print(f"[Warning] {len(missing)} hub IDs in distance files not found in stations.csv: {missing[:10]}...")
except Exception:
    pass

# Load Hub↔Hub distances (km) 
hh = pd.read_csv(HUB_HUB_CSV)
col_h1 = pick_col(hh.columns, 'hubid_from','from','hub_from','h1','k')
col_h2 = pick_col(hh.columns, 'hubid_to','to','hub_to','h2','m')
col_hh = pick_col(hh.columns, 'distance_km','distance','dist','km')
if not (col_h1 and col_h2 and col_hh):
    raise ValueError("HUB_HUB_CSV must have columns like HubID_From, HubID_To, Distance_km")

hh = hh[[col_h1, col_h2, col_hh]].copy()
hh[col_h1] = hh[col_h1].astype(int)
hh[col_h2] = hh[col_h2].astype(int)

# Keep only pairs among the selected hubs
hh = hh[hh[col_h1].isin(H_hubs) & hh[col_h2].isin(H_hubs)]

# Build c_km for all provided hub pairs; if k==m is missing, add 0
c_km = {(int(r[col_h1]), int(r[col_h2])): float(r[col_hh]) for _, r in hh.iterrows()}
for k in H_hubs:
    c_km.setdefault((k,k), 0.0)

# For egress cost c_mj (hub→node), reuse node-hub distances by symmetry: dist(m→j)=dist(j→m)
def cost_mj(m, j):
    return c_ik[(j, m)]  # will KeyError if missing; variable won't be created then

# Sanity: ensure we are not opening more hubs than candidates
if p > len(H_hubs):
    raise ValueError(f"p={p} exceeds the number of candidate hubs ({len(H_hubs)}). Lower p or increase TEST_MAX_HUBS.")

# Cost helpers for readability 
def F_loop(i_hub, k):
    """Cost of routing commodity k through a single hub i (o->i, i->d)."""
    return c_ik[(ok[k], i_hub)] + 0.0 + cost_mj(i_hub, dk[k])

def F_dir(i_hub, j_hub, k):
    """Cost of routing commodity k via two hubs i->j (o->i, i->j, j->d)."""
    return c_ik[(ok[k], i_hub)] + alpha_km * c_km[(i_hub, j_hub)] + cost_mj(j_hub, dk[k])

# Build variable index set A_Ham (with preprocessing) 
A = []  # tuples (i_hub, j_hub, k)
pruned_rev = 0   # removed reverse direction when using undirected cost
pruned_dom = 0   # dominated by loop
added_loops = 0
forced_loops = 0

for k in K_ids:
    o = ok[k]; d = dk[k]

    # Precompute loop costs for hubs that have both o->i and i->d legs available
    loop_cost = {}
    if ALLOW_LOOPS:
        for i in H_hubs:
            if (o, i) in c_ik and (d, i) in c_ik:
                loop_cost[i] = F_loop(i, k)

    # Property 2: if symmetric and o==d, only loops are relevant
    if ALLOW_LOOPS and ASSUME_SYMMETRIC and (o == d):
        for i in loop_cost:
            A.append((i, i, k))
            added_loops += 1
        forced_loops += 1
        continue

    # Consider unordered pairs {i,j}
    H_list = list(H_hubs)
    for idx_i in range(len(H_list)):
        i = H_list[idx_i]
        for idx_j in range(idx_i + 1, len(H_list)):
            j = H_list[idx_j]
            # Feasibility for both directions must exist to compare
            if not ((o, i) in c_ik and (i, j) in c_km and (d, j) in c_ik):
                continue
            if not ((o, j) in c_ik and (j, i) in c_km and (d, i) in c_ik):
                continue

            Fij = F_dir(i, j, k)
            Fji = F_dir(j, i, k)

            if USE_UNDIRECTED:
                # keep only cheaper direction
                if Fij <= Fji:
                    keep_i, keep_j, keep_cost = i, j, Fij
                else:
                    keep_i, keep_j, keep_cost = j, i, Fji
                # Property 1 (dominance by a loop): drop pair if loop cheaper
                if APPLY_DOMINANCE and ALLOW_LOOPS and loop_cost:
                    best_loop = min(loop_cost.get(i, inf), loop_cost.get(j, inf))
                    if keep_cost > best_loop:
                        pruned_dom += 1
                        continue
                A.append((keep_i, keep_j, k))
                pruned_rev += 1  # counted as removing one direction
            else:
                # keep both directions, but still optional dominance test
                if not (APPLY_DOMINANCE and ALLOW_LOOPS and loop_cost and Fij > min(loop_cost.get(i, inf), loop_cost.get(j, inf))):
                    A.append((i, j, k))
                else:
                    pruned_dom += 1
                if not (APPLY_DOMINANCE and ALLOW_LOOPS and loop_cost and Fji > min(loop_cost.get(i, inf), loop_cost.get(j, inf))):
                    A.append((j, i, k))
                else:
                    pruned_dom += 1

    # Optionally include loops alongside pairs when allowed
    if ALLOW_LOOPS:
        for i in loop_cost:
            A.append((i, i, k))
            added_loops += 1

if not A:
    raise RuntimeError("No feasible (i_hub,j_hub,k) tuples after preprocessing. Check CSV consistency or relax flags.")

print(f"Constructed {len(A)} (i_hub,j_hub,k) tuples over |H|={len(H_hubs)} hubs and |K|={len(K_ids)} commodities.")
if USE_UNDIRECTED:
    print(f"[Preprocess] Removed reverse directions: ~{pruned_rev}")
if APPLY_DOMINANCE and ALLOW_LOOPS:
    print(f"[Preprocess] Dropped dominated two-hub pairs: {pruned_dom}")
if ALLOW_LOOPS:
    print(f"[Preprocess] Added loops: {added_loops}; forced-loops commodities (o=d): {forced_loops}")

# Build and solve the model
mdl = Model('p_hub_csv_no_links')
mdl.Params.OutputFlag = 1

# Decision vars
y = mdl.addVars(H_hubs, vtype=GRB.BINARY, name='y')
x = mdl.addVars(A, vtype=GRB.CONTINUOUS, lb=0, ub= 1, name='x')  # Hamacher binary routing

# Objective
mdl.setObjective(
    quicksum(
        wk[k] * ( c_ik[(ok[k], ih)] + alpha_km * c_km[(ih, jh)] + cost_mj(jh, dk[k]) ) * x[(ih, jh, k)]
        for (ih, jh, k) in A
    ), GRB.MINIMIZE
)

# Flow conservation per commodity (Hamacher)
for k in K_ids:
    mdl.addConstr(quicksum(x[(ih, jh, k)] for ih in H_hubs for jh in H_hubs if (ih, jh, k) in x) == 1, name=f"flow_k_{k}")

# Exactly p hubs open
mdl.addConstr(quicksum(y[k] for k in H_hubs) == p, name="hub_limit")

# Hamacher (2004) constraint (3): if a commodity uses hub i (as first or second), hub i must be open
for i in H_hubs:
    for k in K_ids:
        lhs = quicksum(x[(i, j, k)] for j in H_hubs if i != j and (i, j, k) in x)
        lhs += quicksum(x[(j, i, k)] for j in H_hubs if i != j and (j, i, k) in x)
        mdl.addConstr(lhs <= y[i], name=f"hub_usage_i{i}_k{k}")
# Timed solve
t_solve_start = time.perf_counter()
# Optimize
mdl.optimize()
t_solve_end = time.perf_counter()
solve_time_sec = t_solve_end - t_solve_start
# Reporting 
if mdl.Status == GRB.INFEASIBLE:
    print("❌ Model is infeasible.")
    mdl.computeIIS(); mdl.write("infeasible_model.ilp")
else:
    print("\n✅ Objective Function Value:")
    print(f"ObjVal: {mdl.ObjVal}")

# Opened hubs
opened = [k for k in H_hubs if y[k].X > 0.5]
print("\nOpened hubs:", opened)

# Routing decisions (sparse print)
for key, var in x.items():
    if var.X > 1e-8:
        ih, jh, k = key
        print(f"x[i_hub={ih}, j_hub={jh}, k={(ok[k], dk[k])}] = {var.X:.0f}")


# Hub–hub arc usage summary (across all ODs)
from collections import defaultdict as _dd
arc_flow = _dd(float)
for (ih, jh, k), var in x.items():
    if var.X > 1e-9:
        arc_flow[(ih, jh)] += wk[k]

print("\n=== Hub arc connections used (k -> m) ===")
if not arc_flow:
    print("(none)")
else:
    for (k,m), vol in sorted(arc_flow.items()):
        print(f"Hub {k} -> Hub {m}: total_flow = {vol:.6f}")

# Save arcs to CSV for downstream use/plotting
arc_rows = [(k, m, float(vol)) for (k,m), vol in arc_flow.items()]
arc_df = pd.DataFrame(arc_rows, columns=["Hub_From","Hub_To","Total_Flow"])
arc_df.to_csv("hub_arcs.csv", index=False)
print("Saved hub_arcs.csv")


# Simple summary of allocations (origin → first hub), thresholded
origin_to_hub = defaultdict(set)
for (ih, jh, k), var in x.items():
    if var.X > 0.5:
        origin_to_hub[ok[k]].add(ih)

print("\n=== Origin to First Hub Allocation (x > 0.5) ===")
for i in sorted(origin_to_hub):
    print(f"Origin {i} → hubs {sorted(origin_to_hub[i])}")
# After solving the model
output_file = "Results.txt"

with open(output_file, "w") as f:
    # Objective value
    f.write(f" Objective Function Value:\nObjVal: {mdl.ObjVal:.6f}\n")
    f.write(f"Solve time (seconds): {solve_time_sec:.3f}\n\n")
    # Opened hubs
    opened_hubs = [i for i in H_hubs if y[i].X > 0.5]
    f.write(f"Opened hubs: {opened_hubs}\n")

    # x variables > 0.5 (binary in this model)
    for (ih, jh, k), var in x.items():
        if var.X > 0.5:
            f.write(f"x[i_hub={ih}, j_hub={jh}, k=({ok[k]}, {dk[k]})] = {var.X:.0f}\n")

    f.write("\n=== Hub arc connections used (k -> m) ===\n")
    for (k_h, m_h), val in arc_flow.items():
        if val > 1e-6:
            f.write(f"Hub {k_h} -> Hub {m_h}: total_flow = {val:.6f}\n")

    f.write("Saved hub_arcs.csv\n\n")

    # Origin allocations (origin -> first hub) for x > 0.5
    f.write("=== Origin to First Hub Allocation (x > 0.5) ===\n")
    for o in sorted(origin_to_hub):
        hubs_used = sorted(origin_to_hub[o])
        f.write(f"Origin {o} - hubs {hubs_used}\n")

print(f"Detailed results saved to {output_file}")

# Optional plot on India map (Cartopy)
DO_PLOT = True    # generate a map after solving (saved to file)
PLOT_THRESHOLD = 0.5  # draw only strongest flows to reduce plotting load
HUBS_CATALOG_CSV = 'potential_hubs_latlong.csv'  # real 75 hubs with, Latitude/Longitude and names
SAVE_PLOT_PATH = 'Pictorial_representation.png'
SHOW_LABELS = False      # disable labels by default to reduce text rendering
SHOW_LEGEND = True

if DO_PLOT:
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        try:
            from adjustText import adjust_text
            HAVE_ADJUST = True
        except Exception:
            HAVE_ADJUST = False
    except Exception as e:
        print(f"Plot skipped: Cartopy not available ({e})")
    else:
        # Load nodes catalog for coordinates
        nodes = pd.read_csv(NODES_CATALOG_CSV)
        nid = None; nlat = None; nlon = None
        for c in nodes.columns:
            cl = c.lower()
            if cl in ('nodeid','node','id') and nid is None: nid = c
            if cl in ('latitude','lat') and nlat is None: nlat = c
            if cl in ('longitude','lon') and nlon is None: nlon = c
        if not (nid and nlat and nlon):
            print("Plot skipped: nodes_catalog.csv must have NodeID, Latitude, Longitude")
        else:
            # Try to get hub coordinates from a hubs catalog; if not present, fall back to nodes (assuming ids align)
            hubs_cat = None
            try:
                hubs_cat = pd.read_csv(HUBS_CATALOG_CSV)
                # Flexible detection: HubID / Hub_Index / Index / ID; Station name optional
                hid = None; hlat = None; hlon = None; hname = None
                for c in hubs_cat.columns:
                    cl = c.lower()
                    if cl in ('hubid','hub_id','hub_index','index','id') and hid is None: hid = c
                    if cl in ('latitude','lat') and hlat is None: hlat = c
                    if cl in ('longitude','lon') and hlon is None: hlon = c
                    if cl in ('station_name','name') and hname is None: hname = c
                if not (hlat and hlon):
                    hubs_cat = None
                else:
                    if hid is None:
                        hubs_cat = hubs_cat.copy()
                        hubs_cat['HubID'] = np.arange(1, len(hubs_cat)+1)
                        hid = 'HubID'
                    hubs_cat = hubs_cat.rename(columns={hid:'HubID', hlat:'Latitude', hlon:'Longitude'})
                    if hname is not None and 'Name' not in hubs_cat.columns:
                        hubs_cat = hubs_cat.rename(columns={hname:'Name'})
            except Exception:
                hubs_cat = None

            if hubs_cat is None:
                # Fall back: use nodes catalog for hub coordinates (works if HubID space overlaps NodeID space)
                hubs_cat = nodes.rename(columns={nid: 'HubID', nlat: 'Latitude', nlon: 'Longitude'})[['HubID','Latitude','Longitude']]
                print('[Warning] Using nodes_catalog for hubs; check stations.csv columns if hub positions look wrong.')

            fig = plt.figure(figsize=(12,12))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([68, 98, 6, 38], crs=ccrs.PlateCarree())
            # Minimal look: no borders, no coastlines, no axes
            ax.set_axis_off()
            try:
                ax.outline_patch.set_visible(False)
            except Exception:
                pass
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            # Plot hubs: show only TEST_MAX_HUBS candidates as potential (green), opened hubs in red, others light
            candidate_set = set(selected_hubs)
            hubs_candidates = hubs_cat[hubs_cat['HubID'].isin(candidate_set)].copy()
            hubs_open = hubs_candidates[hubs_candidates['HubID'].isin(opened)].copy()
            hubs_potential = hubs_candidates[~hubs_candidates['HubID'].isin(opened)].copy()
            hubs_other = hubs_cat[~hubs_cat['HubID'].isin(candidate_set)].copy()

            # Background: non-candidate hubs (very light)
            if not hubs_other.empty:
                ax.scatter(hubs_other['Longitude'], hubs_other['Latitude'], s=20, c='lightgray', marker='^', edgecolors='k', linewidths=0.2, label='Other Hubs', zorder=3)

            # Potential hubs among the TEST_MAX_HUBS candidates (not opened)
            if not hubs_potential.empty:
                ax.scatter(hubs_potential['Longitude'], hubs_potential['Latitude'], s=40, c='green', marker='^', edgecolors='k', linewidths=0.6, label='Potential Hubs', zorder=5)

            # Opened hubs
            if not hubs_open.empty:
                ax.scatter(hubs_open['Longitude'], hubs_open['Latitude'], s=50, c='red', marker='^', edgecolors='k', linewidths=0.5, label='Opened Hubs', zorder=6)

            # Labels for hubs (optional)
            hub_texts = []
            if SHOW_LABELS:
                for _, r in hubs_open.iterrows():
                    label = (str(r['Name']) if 'Name' in r and pd.notna(r['Name']) else f"H{int(r['HubID'])}")
                    hub_texts.append(ax.text(r['Longitude'], r['Latitude'], label, fontsize=7, transform=ccrs.PlateCarree(), va='bottom', ha='left', zorder=7))

            # Plot OD nodes involved
            used_nodes = sorted(set([i for (i,_) in OD_pairs]) | set([j for (_,j) in OD_pairs]))
            nodes_used = nodes[nodes[nid].isin(used_nodes)].copy()
            nodes_used.rename(columns={nid:'NodeID', nlat:'Latitude', nlon:'Longitude'}, inplace=True)
            ax.scatter(nodes_used['Longitude'], nodes_used['Latitude'], s=12, c='blue', marker='o', alpha=0.6, label='OD Nodes', zorder=3)

            # Node labels (optional, light)
            node_texts = []
            if SHOW_LABELS:
                for _, r in nodes_used.iterrows():
                    node_texts.append(ax.text(r['Longitude'], r['Latitude'], f"N{int(r['NodeID'])}", fontsize=5, transform=ccrs.PlateCarree(), va='bottom', ha='left', zorder=5))

            # Build flows list from binary x_{ih,jh,k}
            from collections import defaultdict
            flows = []
            for (ih, jh, k), var in x.items():
                if var.X >= PLOT_THRESHOLD:
                    flows.append({'o': ok[k], 'd': dk[k], 'i': ih, 'j': jh, 'share': float(var.X)})
            flows_df = pd.DataFrame(flows)
            # Merge coordinates for o, d, i, j
            coord_nodes = nodes.rename(columns={nid:'NodeID', nlat:'Latitude', nlon:'Longitude'})[['NodeID','Latitude','Longitude']]
            coord_hubs  = hubs_cat[['HubID','Latitude','Longitude']].rename(columns={'HubID':'Hub'})

            for _, r in flows_df.iterrows():
                o, d, ih, jh, sh = int(r['o']), int(r['d']), int(r['i']), int(r['j']), float(r['share'])
                pi = coord_nodes[coord_nodes['NodeID'] == o]
                pj = coord_nodes[coord_nodes['NodeID'] == d]
                pk = coord_hubs[coord_hubs['Hub'] == ih]
                pm = coord_hubs[coord_hubs['Hub'] == jh]
                if pi.empty or pj.empty or pk.empty or pm.empty:
                    continue
                lon_i, lat_i = float(pi['Longitude'].iloc[0]), float(pi['Latitude'].iloc[0])
                lon_j, lat_j = float(pj['Longitude'].iloc[0]), float(pj['Latitude'].iloc[0])
                lon_k, lat_k = float(pk['Longitude'].iloc[0]), float(pk['Latitude'].iloc[0])
                lon_m, lat_m = float(pm['Longitude'].iloc[0]), float(pm['Latitude'].iloc[0])
                # Access o->ih (green), optional inter-hub (black if ih!=jh), Egress jh->d (blue)
                ax.plot([lon_i, lon_k], [lat_i, lat_k], color='green', linewidth=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=5)
                if ih != jh:
                    ax.plot([lon_k, lon_m], [lat_k, lat_m], color='black', linewidth=1.5, alpha=0.7, transform=ccrs.PlateCarree(), zorder=5)
                ax.plot([lon_m, lon_j], [lat_m, lat_j], color='blue', linewidth=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=5)

            if HAVE_ADJUST and SHOW_LABELS:
                adjust_text(hub_texts + node_texts, ax=ax, only_move={'points':'y','texts':'y'},
                            arrowprops=dict(arrowstyle='-', color='gray', lw=0.4), expand_text=(1.05,1.05))

            # No legend/title for a clean map-like image
            plt.tight_layout()
            try:
                plt.savefig(SAVE_PLOT_PATH, dpi=300, bbox_inches='tight')
                print(f"Saved plot to {SAVE_PLOT_PATH}")
            except Exception as e:
                print(f"Could not save plot: {e}")
            import matplotlib as _mpl
            # Only show if using an interactive backend
            if not _mpl.get_backend().lower().startswith('agg'):
                plt.show()

# Save results for plotting
# 1) Save opened hubs (IDs only). If you have a hubs catalog with lat/lon, you can merge later.
opened_df = pd.DataFrame({"HubID": opened})
opened_df.to_csv("opened_hubs.csv", index=False)

# 2) Save flows in long form (Origin,Destination,Hub1,Hub2,x,Demand)
flows_rows = []
for (ih, jh, k), var in x.items():
    val = var.X
    if val > 1e-9:
        flows_rows.append((ok[k], dk[k], ih, jh, float(val), wk[k]))
flows_df = pd.DataFrame(flows_rows, columns=["Origin","Destination","Hub1","Hub2","x","Demand"])  # x is 0/1 here
flows_df.to_csv("flows_long.csv", index=False)
print("Saved opened_hubs.csv and flows_long.csv")

# Optional plot on India map (Cartopy)
DO_PLOT = False   # set True to generate a quick map after solving
PLOT_THRESHOLD = 0.5  # show only flows with share >= threshold
HUBS_CATALOG_CSV = 'stations.csv'  # real 75 hubs with Latitude/Longitude and names

if DO_PLOT:
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as e:
        print(f"Plot skipped: Cartopy not available ({e})")
    else:
        # Load nodes catalog for coordinates
        nodes = pd.read_csv(NODES_CATALOG_CSV)
        nid = None; nlat = None; nlon = None
        for c in nodes.columns:
            cl = c.lower()
            if cl in ('nodeid','node','id') and nid is None: nid = c
            if cl in ('latitude','lat') and nlat is None: nlat = c
            if cl in ('longitude','lon') and nlon is None: nlon = c
        if not (nid and nlat and nlon):
            print("Plot skipped: nodes_catalog.csv must have NodeID, Latitude, Longitude")
        else:
            # Try to get hub coordinates from a hubs catalog; if not present, fall back to nodes (assuming ids align)
            hubs_cat = None
            try:
                hubs_cat = pd.read_csv(HUBS_CATALOG_CSV)
                # Flexible detection: HubID / Hub_Index / Index / ID; Station name optional
                hid = None; hlat = None; hlon = None; hname = None
                for c in hubs_cat.columns:
                    cl = c.lower()
                    if cl in ('hubid','hub_id','hub_index','index','id') and hid is None: hid = c
                    if cl in ('latitude','lat') and hlat is None: hlat = c
                    if cl in ('longitude','lon') and hlon is None: hlon = c
                    if cl in ('station_name','name') and hname is None: hname = c
                if not (hlat and hlon):
                    hubs_cat = None
                else:
                    if hid is None:
                        hubs_cat = hubs_cat.copy()
                        hubs_cat['HubID'] = np.arange(1, len(hubs_cat)+1)
                        hid = 'HubID'
                    hubs_cat = hubs_cat.rename(columns={hid:'HubID', hlat:'Latitude', hlon:'Longitude'})
                    if hname is not None and 'Name' not in hubs_cat.columns:
                        hubs_cat = hubs_cat.rename(columns={hname:'Name'})
            except Exception:
                hubs_cat = None

            if hubs_cat is None:
                # Fall back: use nodes catalog for hub coordinates (works if HubID space overlaps NodeID space)
                hubs_cat = nodes.rename(columns={nid: 'HubID', nlat: 'Latitude', nlon: 'Longitude'})[['HubID','Latitude','Longitude']]
                print('[Warning] Using nodes_catalog for hubs; check stations.csv columns if hub positions look wrong.')

            fig = plt.figure(figsize=(14,14))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([68, 98, 6, 38], crs=ccrs.PlateCarree())
            ax.set_axis_off()
            try:
                ax.outline_patch.set_visible(False)
            except Exception:
                pass
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            # Plot hubs: show only TEST_MAX_HUBS candidates as potential (green), opened hubs in red, others light
            candidate_set = set(selected_hubs)
            hubs_candidates = hubs_cat[hubs_cat['HubID'].isin(candidate_set)].copy()
            hubs_open = hubs_candidates[hubs_candidates['HubID'].isin(opened)].copy()
            hubs_potential = hubs_candidates[~hubs_candidates['HubID'].isin(opened)].copy()
            hubs_other = hubs_cat[~hubs_cat['HubID'].isin(candidate_set)].copy()

            if not hubs_other.empty:
                ax.scatter(hubs_other['Longitude'], hubs_other['Latitude'], s=20, c='lightgray', marker='^', edgecolors='k', linewidths=0.2, label='Other Hubs', zorder=3)
            if not hubs_potential.empty:
                ax.scatter(hubs_potential['Longitude'], hubs_potential['Latitude'], s=40, c='green', marker='^', edgecolors='k', linewidths=0.6, label='Potential Hubs', zorder=5)
            if not hubs_open.empty:
                ax.scatter(hubs_open['Longitude'], hubs_open['Latitude'], s=60, c='red', marker='^', edgecolors='k', linewidths=0.5, label='Opened Hubs', zorder=6)

            # Plot OD nodes involved
            used_nodes = sorted(set([i for (i,_) in OD_pairs]) | set([j for (_,j) in OD_pairs]))
            nodes_used = nodes[nodes[nid].isin(used_nodes)].copy()
            nodes_used.rename(columns={nid:'NodeID', nlat:'Latitude', nlon:'Longitude'}, inplace=True)
            ax.scatter(nodes_used['Longitude'], nodes_used['Latitude'], s=12, c='blue', marker='o', alpha=0.6, label='OD Nodes', zorder=3)

            # Draw flows (access, inter-hub, egress) for strong routes
            strong = flows_df[flows_df['x'] >= PLOT_THRESHOLD]
            # Merge coordinates for Origin, Destination, Hub1, Hub2
            coord_nodes = nodes.rename(columns={nid:'NodeID', nlat:'Latitude', nlon:'Longitude'})[['NodeID','Latitude','Longitude']]
            coord_hubs  = hubs_cat[['HubID','Latitude','Longitude']].rename(columns={'HubID':'Hub'})

            for _, r in strong.iterrows():
                o, d, ih, jh, sh = int(r['Origin']), int(r['Destination']), int(r['Hub1']), int(r['Hub2']), float(r['x'])
                # node o
                pi = coord_nodes[coord_nodes['NodeID'] == o]
                pj = coord_nodes[coord_nodes['NodeID'] == d]
                pk = coord_hubs[coord_hubs['Hub'] == ih]
                pm = coord_hubs[coord_hubs['Hub'] == jh]
                if pi.empty or pj.empty or pk.empty or pm.empty:
                    continue
                lon_i, lat_i = float(pi['Longitude'].iloc[0]), float(pi['Latitude'].iloc[0])
                lon_j, lat_j = float(pj['Longitude'].iloc[0]), float(pj['Latitude'].iloc[0])
                lon_k, lat_k = float(pk['Longitude'].iloc[0]), float(pk['Latitude'].iloc[0])
                lon_m, lat_m = float(pm['Longitude'].iloc[0]), float(pm['Latitude'].iloc[0])
                # Access o->ih (green), optional inter-hub (black if ih!=jh), Egress jh->d (blue)
                ax.plot([lon_i, lon_k], [lat_i, lat_k], color='green', linewidth=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=5)
                if ih != jh:
                    ax.plot([lon_k, lon_m], [lat_k, lat_m], color='black', linewidth=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=5)
                ax.plot([lon_m, lon_j], [lat_m, lat_j], color='blue', linewidth=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=5)

            # No legend/title for a clean map-like image
            plt.tight_layout()
            import matplotlib as _mpl
            # Only show if using an interactive backend
            if not _mpl.get_backend().lower().startswith('agg'):
                plt.show()