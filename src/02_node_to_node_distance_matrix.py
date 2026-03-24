import pandas as pd
import numpy as np

# ---- input ----
PATH = "demand_nodes_latlong_OD_clustered.csv"   # your file with Quantity,Start_Lat,Start_Lon,Destination_Lat,Destination_Lon
ROUND_DECIMALS = 5

# ---- haversine (vectorized) in km ----
R = 6371.0
def haversine_km(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

# ---- load ----
df = pd.read_csv(PATH)

# ---- 1) OD distance per row (start→destination) ----
lat_s = np.radians(df["Start_Lat"].values)
lon_s = np.radians(df["Start_Long"].values)
lat_d = np.radians(df["Dest_Lat"].values)
lon_d = np.radians(df["Dest_Long"].values)

od_km = haversine_km(lat_s, lon_s, lat_d, lon_d)
out_od = pd.DataFrame({
    "From_Lat": df["Start_Lat"],
    "From_Lon": df["Start_Long"],
    "To_Lat": df["Dest_Lat"],
    "To_Lon": df["Dest_Long"],
    "Distance_km": np.round(od_km, ROUND_DECIMALS),
    "Quantity": df["Quantity"]
})

# If you prefer an index per OD row:
out_od.index.name = "OD_ID"
out_od.reset_index().to_csv("od_distances_test.csv", index=False)

# ---- 2) (Optional) Node–node long matrix (unique coordinates) ----
# Build a node list from every unique coordinate that appears as start or destination
nodes = pd.concat([
    df[["Start_Lat","Start_Long"]].rename(columns={"Start_Lat":"Latitude","Start_Long":"Longitude"}),
    df[["Dest_Lat","Dest_Long"]].rename(columns={"Dest_Lat":"Latitude","Dest_Long":"Longitude"})
], ignore_index=True).drop_duplicates().reset_index(drop=True)

nodes["NodeID"] = np.arange(1, len(nodes)+1)
nodes = nodes[["NodeID","Latitude","Longitude"]]

# Compute full matrix in long form (may be large!)
lat = np.radians(nodes["Latitude"].values)
lon = np.radians(nodes["Longitude"].values)

# Broadcast to all pairs
lat1 = lat[:, None]
lon1 = lon[:, None]
lat2 = lat[None, :]
lon2 = lon[None, :]

dist_mat = haversine_km(lat1, lon1, lat2, lon2)  # shape (N, N)

# Melt to long form
ii, jj = np.indices(dist_mat.shape)
out_nn = pd.DataFrame({
    "From": nodes["NodeID"].values[ii.ravel()],
    "To": nodes["NodeID"].values[jj.ravel()],
    "Distance_km": np.round(dist_mat.ravel(), ROUND_DECIMALS)
})

# Save both node list and long matrix
nodes.to_csv("nodes_catalog_test.csv", index=False)
out_nn.to_csv("node_node_distance_test.csv", index=False)

print(f"Saved OD distances -> od_distances_test.csv ({len(out_od)} rows)")
print(f"Saved node list -> nodes_catalog_test.csv ({len(nodes)} nodes)")
print(f"Saved node-node distances (long) -> node_node_distance_test.csv ({len(out_nn)} rows)")