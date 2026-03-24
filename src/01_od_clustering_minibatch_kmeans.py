import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

INPUT_CSV = "demand_nodes_latlong.csv"
OUTPUT_CSV = "demand_nodes_latlong_OD_clustered.csv"

N_ORIGIN_CLUSTERS = 20        # <-- adjust freely
N_DEST_CLUSTERS   = 20        # <-- adjust freely
RANDOM_STATE = 42

# LOAD DATA
df = pd.read_csv(INPUT_CSV)
df = df.rename(columns={
    "Quantity": "Demand",
    "Start_Lat": "o_lat",
    "Start_Long": "o_lon",
    "Dest_Lat": "d_lat",
    "Dest_Long": "d_lon"
})

print("Loaded rows:", len(df))

# CLUSTER ORIGINS
origin_features = df[["o_lat", "o_lon"]].values

origin_kmeans = MiniBatchKMeans(
    n_clusters=N_ORIGIN_CLUSTERS,
    batch_size=10000,
    random_state=RANDOM_STATE
)
origin_kmeans.fit(origin_features, sample_weight=df["Demand"])
df["origin_cluster"] = origin_kmeans.predict(origin_features)
origin_centroids = origin_kmeans.cluster_centers_
print("Origin clustering done.")

# CLUSTER DESTINATIONS
dest_features = df[["d_lat", "d_lon"]].values
dest_kmeans = MiniBatchKMeans(
    n_clusters=N_DEST_CLUSTERS,
    batch_size=10000,
    random_state=RANDOM_STATE
)
dest_kmeans.fit(dest_features, sample_weight=df["Demand"])
df["dest_cluster"] = dest_kmeans.predict(dest_features)
dest_centroids = dest_kmeans.cluster_centers_
print("Destination clustering done.")

# AGGREGATE FLOWS BETWEEN CLUSTERS
agg = (
    df.groupby(["origin_cluster", "dest_cluster"], as_index=False)
      .agg({"Demand": "sum"})
)
print("Aggregation complete.")
print("Reduced OD count:", len(agg))

# Map centroid coordinates
agg["Start_Lat"] = agg["origin_cluster"].apply(lambda x: origin_centroids[x][0])
agg["Start_Long"] = agg["origin_cluster"].apply(lambda x: origin_centroids[x][1])
agg["Dest_Lat"] = agg["dest_cluster"].apply(lambda x: dest_centroids[x][0])
agg["Dest_Long"] = agg["dest_cluster"].apply(lambda x: dest_centroids[x][1])

# Final format
final_df = agg[["Demand", "Start_Lat", "Start_Long", "Dest_Lat", "Dest_Long"]]
final_df = final_df.rename(columns={"Demand": "Quantity"})
final_df.to_csv(OUTPUT_CSV, index=False)
print("Clustered OD file saved:", OUTPUT_CSV)

# PLOT ORIGINAL VS CLUSTERED
plt.figure(figsize=(10, 10))

# Original origins (light blue)
plt.scatter(df["o_lon"], df["o_lat"], s=5, alpha=0.2)

# Original destinations (light green)
plt.scatter(df["d_lon"], df["d_lat"], s=5, alpha=0.2)

# Clustered origin centroids (red)
plt.scatter(origin_centroids[:,1], origin_centroids[:,0], s=200, marker='^')

# Clustered destination centroids (black)
plt.scatter(dest_centroids[:,1], dest_centroids[:,0], s=200, marker='s')

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Original vs Clustered OD Nodes")
plt.tight_layout()
plt.savefig("clustered_map.png", dpi=300)
plt.close()
print("Map saved as clustered_map.png")