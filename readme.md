# Intermodal Hub Location Optimization

This repository provides an optimization framework for solving intermodal hub location problems in transportation systems.

## 🔧 Workflow

The workflow consists of the following steps:

1. **Clustering**
   - MiniBatch K-Means to reduce OD data
   - File: `01_od_clustering_minibatch_kmeans.py`

2. **Distance Matrix (Node-Node)**
   - Computes distances using latitude and longitude
   - File: `02_node_to_node_distance_matrix.py`

3. **Hub Distance Matrix**
   - Node-to-hub and hub-to-hub distances
   - File: `03_hub_distance_matrix.py`

4. **P-Hub Median Model**
   - Mathematical formulation for hub selection
   - File: `04_p_hub_median_model.py`

5. **Two-Phase Benders Decomposition**
   - Advanced solution method for large-scale problems
   - File: `05_two_phase_benders_decomposition.py`

## 📂 Input Data

- `demand_nodes_latlong.csv`
- `potential_hubs_latlong.csv`

## ▶️ Run Order

Run the scripts in the following order: 1 → 2 → 3 → 4 → 5

## ⚙️ Requirements

- Python 3.x
- Gurobi Optimizer
- NumPy, Pandas, Matplotlib, Scikit-learn

## 📌 Applications

- Rail transportation planning
- Hub network design
- Logistics and supply chain optimization

---

## 👤 Author

Chanchal Kumar Salode  
PhD, IIT Delhi  
