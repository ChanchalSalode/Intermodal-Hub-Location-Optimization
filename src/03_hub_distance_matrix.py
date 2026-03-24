import pandas as pd
import numpy as np

# -------- inputs --------
OD_PATH   = "demand_nodes_latlong_OD_clustered.csv"
HUBS_PATH = "potential_hubs_latlong.csv"
ROUND_DEC = 5

# -------- haversine (km) --------
R = 6371.0
def haversine_km(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

# -------- load OD nodes (unique) --------
od = pd.read_csv(OD_PATH)

nodes = pd.concat([
    od[["Start_Lat","Start_Long"]].rename(columns={"Start_Lat":"Latitude","Start_Long":"Longitude"}),
    od[["Dest_Lat","Dest_Long"]].rename(columns={"Dest_Lat":"Latitude","Dest_Long":"Longitude"}),
], ignore_index=True).drop_duplicates().reset_index(drop=True)

if "NodeID" not in nodes.columns:
    nodes["NodeID"] = np.arange(1, len(nodes)+1)

nodes = nodes[["NodeID","Latitude","Longitude"]]

# -------- load hubs --------
hubs = pd.read_csv(HUBS_PATH)
if "HubID" not in hubs.columns:
    hubs["HubID"] = np.arange(1, len(hubs)+1)
hubs = hubs[["HubID","Latitude","Longitude"]]

# -------- vectorize angles --------
n_lat = np.radians(nodes["Latitude"].values)
n_lon = np.radians(nodes["Longitude"].values)
h_lat = np.radians(hubs["Latitude"].values)
h_lon = np.radians(hubs["Longitude"].values)

# -------- Node -> Hub distances (long format) --------
# broadcast: nodes (N x 1) vs hubs (1 x H)
n_lat_col = n_lat[:, None]
n_lon_col = n_lon[:, None]
h_lat_row = h_lat[None, :]
h_lon_row = h_lon[None, :]

node_hub_km = haversine_km(n_lat_col, n_lon_col, h_lat_row, h_lon_row)  # shape (N, H)

I, J = np.indices(node_hub_km.shape)
node_to_hub = pd.DataFrame({
    "NodeID":   nodes["NodeID"].values[I.ravel()],
    "HubID":    hubs["HubID"].values[J.ravel()],
    "Distance_km": np.round(node_hub_km.ravel(), ROUND_DEC)
})

node_to_hub.to_csv("node_to_hub_distance_test.csv", index=False)

# -------- Hub <-> Hub distances (long format) --------
h_lat_col = h_lat[:, None]
h_lon_col = h_lon[:, None]
h_lat_row = h_lat[None, :]
h_lon_row = h_lon[None, :]

hub_hub_km = haversine_km(h_lat_col, h_lon_col, h_lat_row, h_lon_row)  # (H, H)
A, B = np.indices(hub_hub_km.shape)
hub_to_hub = pd.DataFrame({
    "HubID_From": hubs["HubID"].values[A.ravel()],
    "HubID_To":   hubs["HubID"].values[B.ravel()],
    "Distance_km": np.round(hub_hub_km.ravel(), ROUND_DEC)
})

hub_to_hub.to_csv("hub_to_hub_distance_test.csv", index=False)

# -------- little summary --------
print(f"Nodes: {len(nodes)}  |  Hubs: {len(hubs)}")
print(f"Saved Node→Hub distances: node_to_hub_distance_test.csv  (rows = {len(node_to_hub)})")
print(f"Saved Hub↔Hub distances:  hub_to_hub_distance_test.csv   (rows = {len(hub_to_hub)})")