import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def perform_step2_feature_encoding():
    print("==================================================")
    print("STEP 2: FEATURE ENCODING REPORT")
    print("==================================================\n")

    input_clients_path = os.path.join("data", "cleaned_clients.csv")
    output_encoded_path = os.path.join("data", "encoded_clients.csv")

    if not os.path.exists(input_clients_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {input_clients_path}. Run Step 1 first.")

    clients_df = pd.read_csv(input_clients_path)
    encoded_df = clients_df.copy()

    # Target categorical columns for Step 2
    target_cols = ['client_type', 'region', 'acquisition_purpose', 'referral_channel', 'country']

    print("--- 1. CATEGORICAL COLUMNS BEFORE ENCODING ---")
    for col in target_cols:
        unique_vals = clients_df[col].unique()
        print(f"  - {col} ({len(unique_vals)} unique values): Sample = {list(unique_vals[:5])}")
    print()

    # ----------------------------------------------------
    # ENCODING DECISIONS & IMPLEMENTATION
    # ----------------------------------------------------
    print("--- 2. ENCODING METHOD SELECTION & RATIONALE ---")

    # A. client_type: Binary Nominal -> Label Encoding (0/1)
    # Rationale: Exactly 2 categories ('Individual': 0, 'Company': 1). Binary mapping avoids extra columns.
    client_type_map = {'Individual': 0, 'Company': 1}
    encoded_df['client_type_encoded'] = encoded_df['client_type'].map(client_type_map)
    print("  [1] client_type -> Binary Label Encoding ('Individual': 0, 'Company': 1)")

    # B. acquisition_purpose: Binary Nominal -> Label Encoding (0/1)
    # Rationale: Exactly 2 categories ('Home': 0, 'Investment': 1). Binary mapping preserves compactness.
    purpose_map = {'Home': 0, 'Investment': 1}
    encoded_df['acquisition_purpose_encoded'] = encoded_df['acquisition_purpose'].map(purpose_map)
    print("  [2] acquisition_purpose -> Binary Label Encoding ('Home': 0, 'Investment': 1)")

    # C. referral_channel: Multi-class Nominal (3 unique) -> One-Hot Encoding
    # Rationale: 3 nominal categories ('Website', 'Agency', 'Client') without natural ordering. One-hot encoding prevents arbitrary rank assumptions.
    referral_dummies = pd.get_dummies(encoded_df['referral_channel'], prefix='referral_channel', dtype=int)
    encoded_df = pd.concat([encoded_df, referral_dummies], axis=1)
    print(f"  [3] referral_channel -> One-Hot Encoding ({len(referral_dummies.columns)} columns generated)")

    # D. country: Multi-class Nominal (10 unique) -> One-Hot Encoding
    # Rationale: 10 geographical country categories without intrinsic order. One-hot encoding models each country independently.
    country_dummies = pd.get_dummies(encoded_df['country'], prefix='country', dtype=int)
    encoded_df = pd.concat([encoded_df, country_dummies], axis=1)
    print(f"  [4] country -> One-Hot Encoding ({len(country_dummies.columns)} columns generated)")

    # E. region: High-Cardinality Nominal (57 unique) -> Label Encoding
    # Rationale: 57 distinct regions. Label encoding is used to keep feature dimensionality manageable and prevent severe matrix sparsity.
    le_region = LabelEncoder()
    encoded_df['region_encoded'] = le_region.fit_transform(encoded_df['region'])
    print("  [5] region -> Label Encoding (57 unique regions mapped to integers 0-56 to control dimensionality)")
    print()

    # ----------------------------------------------------
    # 3. VERIFICATION & ENCODED COLUMNS REPORT
    # ----------------------------------------------------
    new_encoded_cols = (
        ['client_type_encoded', 'acquisition_purpose_encoded', 'region_encoded'] +
        list(referral_dummies.columns) +
        list(country_dummies.columns)
    )

    print("--- 3. RESULTING ENCODED COLUMNS ---")
    for col in new_encoded_cols:
        print(f"  - {col} (dtype: {encoded_df[col].dtype}): Min={encoded_df[col].min()}, Max={encoded_df[col].max()}")
    print()

    # 4. VERIFY NO STRING VALUES IN SELECTED VARIABLES
    non_numeric_encoded = [col for col in new_encoded_cols if not pd.api.types.is_numeric_dtype(encoded_df[col])]
    
    print("--- 4. DATA TYPE VERIFICATION ---")
    if len(non_numeric_encoded) == 0:
        print("  [SUCCESS] All target encoded variables are purely numeric (int64). Zero string values found.")
    else:
        print(f"  [WARNING] Non-numeric encoded columns detected: {non_numeric_encoded}")
    print()

    # Save encoded dataset
    encoded_df.to_csv(output_encoded_path, index=False)
    print(f"--- 5. OUTPUT SAVED ---")
    print(f"  - Encoded dataset saved to: {output_encoded_path}")
    print(f"  - Total Dataset Columns: {len(encoded_df.columns)} (Original 12 + Encoded features)")
    print("\nStep 2 Feature Encoding completed successfully.")

if __name__ == "__main__":
    perform_step2_feature_encoding()
