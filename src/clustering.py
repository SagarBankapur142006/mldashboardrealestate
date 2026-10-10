import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

def perform_step4_clustering():
    print("==================================================")
    print("STEP 4: CLUSTERING MODEL SELECTION REPORT")
    print("==================================================\n")

    input_scaled_path = os.path.join("data", "scaled_clients.csv")
    output_clustered_path = os.path.join("data", "clustered_clients.csv")

    if not os.path.exists(input_scaled_path):
        raise FileNotFoundError(f"Scaled dataset not found at {input_scaled_path}. Run Step 3 first.")

    clients_df = pd.read_csv(input_scaled_path)
    clustered_df = clients_df.copy()

    # ----------------------------------------------------
    # 1. PREPARE CLUSTERING FEATURE MATRIX
    # ----------------------------------------------------
    # Exclude: client_id, PII (first_name, last_name), unencoded strings,
    # and unscaled numerical features (age, satisfaction_score)
    feature_cols = [
        "client_type_encoded",
        "acquisition_purpose_encoded",
        "region_encoded",
        "referral_channel_Agency",
        "referral_channel_Client",
        "referral_channel_Website",
        "country_Australia",
        "country_Belgium",
        "country_Canada",
        "country_Denmark",
        "country_France",
        "country_Germany",
        "country_Mexico",
        "country_Russia",
        "country_UK",
        "country_USA",
        "age_scaled",
        "satisfaction_score_scaled"
    ]

    # Include binary loan_applied_encoded if loan_applied exists
    if "loan_applied" in clustered_df.columns and "loan_applied_encoded" not in clustered_df.columns:
        clustered_df["loan_applied_encoded"] = (clustered_df["loan_applied"] == "Yes").astype(int)
        feature_cols.append("loan_applied_encoded")

    X = clustered_df[feature_cols].copy()

    # Integrity Checks
    null_count = X.isnull().sum().sum()
    inf_count = np.isinf(X).sum().sum()
    non_numeric = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(X[col])]

    print("--- 1. FEATURE MATRIX INTEGRITY CHECKS ---")
    print(f"  - Total Records: {len(X)}")
    print(f"  - Total Features Selected: {len(feature_cols)}")
    print(f"  - Missing Values (NaN): {null_count}")
    print(f"  - Infinite Values (Inf): {inf_count}")
    print(f"  - Non-Numeric Features: {len(non_numeric)}")
    
    if null_count == 0 and inf_count == 0 and len(non_numeric) == 0:
        print("  [SUCCESS] Feature matrix is clean, numeric, and fully prepared for clustering.\n")
    else:
        raise ValueError("Feature matrix contains invalid values.")

    # ----------------------------------------------------
    # 2. MODEL 1: K-MEANS CLUSTERING (K = 4)
    # ----------------------------------------------------
    n_clusters = 4
    random_state = 42

    print(f"--- 2. FITTING K-MEANS CLUSTERING (K = {n_clusters}, random_state = {random_state}) ---")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    clustered_df["kmeans_cluster"] = kmeans_labels

    kmeans_counts = pd.Series(kmeans_labels).value_counts().sort_index()
    print("  K-Means Cluster Sizes:")
    for cluster_id, count in kmeans_counts.items():
        percentage = (count / len(X)) * 100
        print(f"    - Cluster {cluster_id}: {count} buyers ({percentage:.2f}%)")
    print()

    # ----------------------------------------------------
    # 3. MODEL 2: HIERARCHICAL AGGLOMERATIVE CLUSTERING (K = 4)
    # ----------------------------------------------------
    # Linkage Configuration Rationale:
    # 'ward' linkage minimizes total within-cluster variance.
    # It requires Euclidean distance metric and works effectively on standardized numerical feature matrices.
    print(f"--- 3. FITTING HIERARCHICAL CLUSTERING (K = {n_clusters}, linkage = 'ward', metric = 'euclidean') ---")
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    agg_labels = agg_clustering.fit_predict(X)
    clustered_df["hierarchical_cluster"] = agg_labels

    agg_counts = pd.Series(agg_labels).value_counts().sort_index()
    print("  Hierarchical Cluster Sizes:")
    for cluster_id, count in agg_counts.items():
        percentage = (count / len(X)) * 100
        print(f"    - Cluster {cluster_id}: {count} buyers ({percentage:.2f}%)")
    print()

    # ----------------------------------------------------
    # 4. SAVE CLUSTERED DATASET
    # ----------------------------------------------------
    clustered_df.to_csv(output_clustered_path, index=False)

    print("--- 4. OUTPUT SAVED ---")
    print(f"  - Clustered dataset saved to: {output_clustered_path}")
    print(f"  - Total Dataset Columns: {len(clustered_df.columns)}")
    print(f"  - Saved Cluster Assignments: 'kmeans_cluster' & 'hierarchical_cluster'")
    print("\nStep 4 Clustering Model Selection completed successfully.")

if __name__ == "__main__":
    perform_step4_clustering()
