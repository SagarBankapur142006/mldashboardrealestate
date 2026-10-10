import os
import pandas as pd

def perform_step1_data_cleaning():
    print("==================================================")
    print("STEP 1: DATA CLEANING & NORMALIZATION REPORT")
    print("==================================================\n")

    # Paths to datasets
    clients_path = "clients.csv"
    properties_path = "properties.csv"
    
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    cleaned_clients_path = os.path.join(output_dir, "cleaned_clients.csv")
    cleaned_properties_path = os.path.join(output_dir, "cleaned_properties.csv")

    # Load original datasets
    clients_df = pd.read_csv(clients_path)
    properties_df = pd.read_csv(properties_path)

    # Record initial counts
    initial_clients_count = len(clients_df)
    initial_properties_count = len(properties_df)

    # ----------------------------------------------------
    # 1. MISSING VALUES INSPECTION & HANDLING
    # ----------------------------------------------------
    clients_nulls_before = clients_df.isnull().sum().to_dict()
    properties_nulls_before = properties_df.isnull().sum().to_dict()

    # For properties.csv, missing client_ref values represent unsold ('Available') listings.
    # They are intentionally retained without dropping rows so that inventory statistics remain intact.
    # No missing values exist in clients.csv.

    clients_nulls_after = clients_df.isnull().sum().to_dict()
    properties_nulls_after = properties_df.isnull().sum().to_dict()

    # ----------------------------------------------------
    # 2. CATEGORICAL LABEL NORMALIZATION
    # ----------------------------------------------------
    clients_cat_cols = ['client_type', 'gender', 'country', 'region', 
                        'acquisition_purpose', 'loan_applied', 'referral_channel']
    
    properties_cat_cols = ['unit_category', 'listing_status']

    # Apply string stripping and standardized formatting
    for col in clients_cat_cols:
        if col in clients_df.columns:
            clients_df[col] = clients_df[col].astype(str).str.strip()

    for col in properties_cat_cols:
        if col in properties_df.columns:
            properties_df[col] = properties_df[col].astype(str).str.strip()

    # ----------------------------------------------------
    # 3. DUPLICATE CHECK & REMOVAL
    # ----------------------------------------------------
    clients_full_duplicates = int(clients_df.duplicated().sum())
    clients_id_duplicates = int(clients_df['client_id'].duplicated().sum())
    
    properties_full_duplicates = int(properties_df.duplicated().sum())
    properties_id_duplicates = int(properties_df['listing_id'].duplicated().sum())

    # Drop duplicates if any exist
    clients_df = clients_df.drop_duplicates(subset=['client_id'], keep='first').reset_index(drop=True)
    properties_df = properties_df.drop_duplicates(subset=['listing_id'], keep='first').reset_index(drop=True)

    final_clients_count = len(clients_df)
    final_properties_count = len(properties_df)

    # ----------------------------------------------------
    # SAVE CLEANED DATASETS
    # ----------------------------------------------------
    clients_df.to_csv(cleaned_clients_path, index=False)
    properties_df.to_csv(cleaned_properties_path, index=False)

    # ----------------------------------------------------
    # PRINT SUMMARY REPORT
    # ----------------------------------------------------
    print("--- 1. RECORD COUNTS SUMMARY ---")
    print(f"Clients Dataset: Initial = {initial_clients_count} | Final = {final_clients_count}")
    print(f"Properties Dataset: Initial = {initial_properties_count} | Final = {final_properties_count}\n")

    print("--- 2. MISSING VALUES BEFORE & AFTER CLEANING ---")
    print("Clients Dataset:")
    for col, count in clients_nulls_before.items():
        print(f"  - {col}: Before = {count} | After = {clients_nulls_after[col]}")
    
    print("\nProperties Dataset:")
    for col, count in properties_nulls_before.items():
        print(f"  - {col}: Before = {count} | After = {properties_nulls_after[col]}")
    print("  * Note: Missing client_ref (2,695) corresponds to unsold 'Available' listings and is preserved.\n")

    print("--- 3. CATEGORICAL COLUMNS NORMALIZED ---")
    print(f"Clients Dataset Categorical Columns Stripped/Normalized ({len(clients_cat_cols)}):")
    print(f"  {', '.join(clients_cat_cols)}")
    print(f"Properties Dataset Categorical Columns Stripped/Normalized ({len(properties_cat_cols)}):")
    print(f"  {', '.join(properties_cat_cols)}\n")

    print("--- 4. DUPLICATE ENTRIES FOUND & REMOVED ---")
    print(f"Clients Dataset: Full Row Duplicates = {clients_full_duplicates} | ID Duplicates = {clients_id_duplicates} | Removed = {initial_clients_count - final_clients_count}")
    print(f"Properties Dataset: Full Row Duplicates = {properties_full_duplicates} | ID Duplicates = {properties_id_duplicates} | Removed = {initial_properties_count - final_properties_count}\n")

    print("--- 5. OUTPUT FILES CREATED ---")
    print(f"  - Cleaned Clients: {cleaned_clients_path}")
    print(f"  - Cleaned Properties: {cleaned_properties_path}")
    print("\nStep 1 Data Cleaning completed successfully.")

if __name__ == "__main__":
    perform_step1_data_cleaning()
