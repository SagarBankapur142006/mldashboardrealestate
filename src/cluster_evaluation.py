import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def perform_step5_cluster_evaluation():
    print("==================================================")
    print("STEP 5: OPTIMAL CLUSTER SELECTION REPORT")
    print("==================================================\n")

    input_scaled_path = os.path.join("data", "scaled_clients.csv")
    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")
    
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if not os.path.exists(input_scaled_path):
        raise FileNotFoundError(f"Scaled dataset not found at {input_scaled_path}. Run Step 3 first.")

    clients_df = pd.read_csv(input_scaled_path)

    # Prepare feature matrix (same 19 features as Step 4)
    if "loan_applied_encoded" not in clients_df.columns:
        clients_df["loan_applied_encoded"] = (clients_df["loan_applied"] == "Yes").astype(int)

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
        "satisfaction_score_scaled",
        "loan_applied_encoded"
    ]

    X = clients_df[feature_cols].copy()

    # ----------------------------------------------------
    # 1. EVALUATE K-MEANS FOR K = 2 TO 10
    # ----------------------------------------------------
    k_range = range(2, 11)
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        inertia = kmeans.inertia_
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        results.append({
            "K": k,
            "Inertia": inertia,
            "Silhouette_Score": sil,
            "Calinski_Harabasz": ch,
            "Davies_Bouldin": db
        })

    eval_df = pd.DataFrame(results)

    # Save evaluation table to CSV
    eval_csv_path = os.path.join(reports_dir, "cluster_evaluation.csv")
    eval_df.to_csv(eval_csv_path, index=False)

    print("--- 1. CLUSTER EVALUATION METRICS TABLE (K = 2 to 10) ---")
    print(eval_df.to_string(index=False))
    print()

    # ----------------------------------------------------
    # 2. GENERATE PLOTS
    # ----------------------------------------------------
    sns.set_theme(style="whitegrid")

    # Plot 1: Elbow Method
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_df["K"], eval_df["Inertia"], marker="o", linewidth=2.5, color="#1f77b4", markersize=8)
    ax.set_title("Elbow Method for Optimal K (Inertia vs. Number of Clusters)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    ax.set_xticks(list(k_range))
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Highlight K=4
    k4_inertia = eval_df.loc[eval_df["K"] == 4, "Inertia"].values[0]
    ax.plot(4, k4_inertia, marker="o", color="red", markersize=12, label="Provisional K=4")
    ax.legend()
    plt.tight_layout()

    elbow_png_path = os.path.join(reports_dir, "elbow_method.png")
    fig.savefig(elbow_png_path, dpi=300)
    fig.savefig(os.path.join(figures_dir, "elbow_method.png"), dpi=300)
    plt.close(fig)

    # Plot 2: Silhouette Score
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_df["K"], eval_df["Silhouette_Score"], marker="s", linewidth=2.5, color="#2ca02c", markersize=8)
    ax.set_title("Silhouette Score vs. Number of Clusters (K)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_xticks(list(k_range))
    ax.grid(True, linestyle="--", alpha=0.7)

    # Highlight max silhouette score and K=4
    max_sil_row = eval_df.loc[eval_df["Silhouette_Score"].idxmax()]
    ax.plot(max_sil_row["K"], max_sil_row["Silhouette_Score"], marker="*", color="gold", markersize=16, label=f"Max Silhouette (K={int(max_sil_row['K'])})")
    ax.plot(4, eval_df.loc[eval_df["K"] == 4, "Silhouette_Score"].values[0], marker="o", color="red", markersize=10, label="Provisional K=4")
    ax.legend()
    plt.tight_layout()

    sil_png_path = os.path.join(reports_dir, "silhouette_scores.png")
    fig.savefig(sil_png_path, dpi=300)
    fig.savefig(os.path.join(figures_dir, "silhouette_scores.png"), dpi=300)
    plt.close(fig)

    # ----------------------------------------------------
    # 3. ANALYSIS & RECOMMENDATION
    # ----------------------------------------------------
    max_k = int(max_sil_row["K"])
    max_sil_val = max_sil_row["Silhouette_Score"]
    k4_sil_val = eval_df.loc[eval_df["K"] == 4, "Silhouette_Score"].values[0]

    # Calculate percentage change in inertia (Elbow analysis)
    eval_df["Inertia_Diff"] = eval_df["Inertia"].diff().abs()
    eval_df["Inertia_Drop_%"] = (eval_df["Inertia_Diff"] / eval_df["Inertia"].shift(1)) * 100

    print("--- 2. ANALYSIS OF FINDINGS ---")
    print(f"  - Peak Silhouette Score K: K = {max_k} (Score = {max_sil_val:.4f})")
    print(f"  - Provisional K = 4 Silhouette Score: Score = {k4_sil_val:.4f}")
    print("  - Inertia Drops (% reduction per K increment):")
    for idx, row in eval_df.iterrows():
        if not np.isnan(row["Inertia_Drop_%"]):
            print(f"      K={int(row['K']-1)} -> K={int(row['K'])}: Inertia decreased by {row['Inertia_Drop_%']:.2f}%")
    print()

    print("--- 3. CLUSTER RECOMMENDATION & SYNTHESIS ---")
    print(f"  - Recommended K: K = {max_k} (or K = 4 for domain alignment)")
    print(f"  - Explanation: Silhouette score reaches its global maximum at K = {max_k} ({max_sil_val:.4f}).")
    print("  - Inertia drop flattens noticeably around K = 4 to K = 5.")
    print("  - Provisional dataset data/clustered_clients.csv remains unchanged.\n")

    print("--- 4. OUTPUT FILES CREATED ---")
    print(f"  - Metrics Table CSV: {eval_csv_path}")
    print(f"  - Elbow Plot: {elbow_png_path}")
    print(f"  - Silhouette Plot: {sil_png_path}")
    print("\nStep 5 Cluster Evaluation completed successfully.")

if __name__ == "__main__":
    perform_step5_cluster_evaluation()
