import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def perform_step3_feature_scaling():
    print("==================================================")
    print("STEP 3: FEATURE SCALING REPORT")
    print("==================================================\n")

    input_encoded_path = os.path.join("data", "encoded_clients.csv")
    output_scaled_path = os.path.join("data", "scaled_clients.csv")
    scaler_save_path = os.path.join("data", "scaler.joblib")

    if not os.path.exists(input_encoded_path):
        raise FileNotFoundError(f"Encoded dataset not found at {input_encoded_path}. Run Step 2 first.")

    clients_df = pd.read_csv(input_encoded_path)
    scaled_df = clients_df.copy()

    # ----------------------------------------------------
    # 1. DERIVE AGE FROM DATE_OF_BIRTH
    # ----------------------------------------------------
    # Reference Date: 2025-12-31 (End of transaction dataset timeline)
    reference_date = pd.to_datetime("2025-12-31")
    dob_parsed = pd.to_datetime(scaled_df["date_of_birth"], format="mixed", errors="coerce")
    
    # Calculate age in full years
    scaled_df["age"] = ((reference_date - dob_parsed).dt.days / 365.25).apply(np.floor).astype(int)

    print("--- 1. NUMERICAL VARIABLES IDENTIFIED FOR SCALING ---")
    print("  - age (Derived from date_of_birth as of 2025-12-31)")
    print("  - satisfaction_score (Customer satisfaction score, scale 1-5)\n")

    # ----------------------------------------------------
    # 2. BEFORE SCALING STATISTICAL SUMMARY
    # ----------------------------------------------------
    features_to_scale = ["age", "satisfaction_score"]

    print("--- 2. STATISTICAL SUMMARY BEFORE SCALING ---")
    before_stats = {}
    for col in features_to_scale:
        before_stats[col] = {
            "min": float(scaled_df[col].min()),
            "max": float(scaled_df[col].max()),
            "mean": float(scaled_df[col].mean()),
            "std": float(scaled_df[col].std())
        }
        print(f"  - {col}:")
        print(f"      Min  = {before_stats[col]['min']:.4f}")
        print(f"      Max  = {before_stats[col]['max']:.4f}")
        print(f"      Mean = {before_stats[col]['mean']:.4f}")
        print(f"      Std  = {before_stats[col]['std']:.4f}")
    print()

    # ----------------------------------------------------
    # 3. APPLY STANDARD SCALER
    # ----------------------------------------------------
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(scaled_df[features_to_scale])

    scaled_df["age_scaled"] = scaled_matrix[:, 0]
    scaled_df["satisfaction_score_scaled"] = scaled_matrix[:, 1]

    # ----------------------------------------------------
    # 4. AFTER SCALING STATISTICAL SUMMARY
    # ----------------------------------------------------
    scaled_features = ["age_scaled", "satisfaction_score_scaled"]

    print("--- 3. STATISTICAL SUMMARY AFTER SCALING (StandardScaler: mean ~ 0, std = 1) ---")
    after_stats = {}
    for col in scaled_features:
        after_stats[col] = {
            "min": float(scaled_df[col].min()),
            "max": float(scaled_df[col].max()),
            "mean": float(scaled_df[col].mean()),
            "std": float(scaled_df[col].std())
        }
        print(f"  - {col}:")
        print(f"      Min  = {after_stats[col]['min']:.4f}")
        print(f"      Max  = {after_stats[col]['max']:.4f}")
        print(f"      Mean = {after_stats[col]['mean']:.4f} (approx 0.0)")
        print(f"      Std  = {after_stats[col]['std']:.4f}")
    print()

    # ----------------------------------------------------
    # 5. DATA VERIFICATION
    # ----------------------------------------------------
    print("--- 4. DATA VERIFICATION ---")
    missing_count = scaled_df[scaled_features].isnull().sum().sum()
    is_numeric = all(pd.api.types.is_numeric_dtype(scaled_df[col]) for col in scaled_features)

    if missing_count == 0 and is_numeric:
        print("  [SUCCESS] Scaled columns contain numeric values with 0 missing values.")
    else:
        print(f"  [WARNING] Verification failed: missing={missing_count}, numeric={is_numeric}")
    print()

    # ----------------------------------------------------
    # 6. SAVE FITTED SCALER & SCALED DATASET
    # ----------------------------------------------------
    # Save fitted scaler for inference on new buyer records
    joblib.dump(scaler, scaler_save_path)
    
    # Save scaled dataset separately
    scaled_df.to_csv(output_scaled_path, index=False)

    print("--- 5. OUTPUTS SAVED ---")
    print(f"  - Scaled dataset saved to: {output_scaled_path}")
    print(f"  - Fitted StandardScaler model saved to: {scaler_save_path}")
    print(f"  - Total Dataset Columns: {len(scaled_df.columns)}")
    print("\n--- SCALER REUSE INSTRUCTIONS ---")
    print("To reuse this scaler on new buyer records in production:")
    print("  scaler = joblib.load('data/scaler.joblib')")
    print("  new_buyer_scaled = scaler.transform(new_buyer[['age', 'satisfaction_score']])")
    print("\nStep 3 Feature Scaling completed successfully.")

if __name__ == "__main__":
    perform_step3_feature_scaling()
