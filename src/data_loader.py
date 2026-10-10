import os
import pandas as pd
from typing import Tuple, Dict, Any
from src.utils import RAW_CLIENTS_PATH, RAW_PROPERTIES_PATH

class DataLoader:
    """Class responsible for loading, validating, and managing raw real estate datasets."""

    def __init__(self, clients_path: str = RAW_CLIENTS_PATH, properties_path: str = RAW_PROPERTIES_PATH):
        self.clients_path = clients_path
        self.properties_path = properties_path
        self.clients_df = None
        self.properties_df = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads clients and properties datasets from CSV files with error handling."""
        if not os.path.exists(self.clients_path):
            raise FileNotFoundError(f"Clients dataset not found at: {self.clients_path}")
        if not os.path.exists(self.properties_path):
            raise FileNotFoundError(f"Properties dataset not found at: {self.properties_path}")

        try:
            self.clients_df = pd.read_csv(self.clients_path)
            self.properties_df = pd.read_csv(self.properties_path)
        except Exception as e:
            raise RuntimeError(f"Error reading CSV datasets: {str(e)}")

        return self.clients_df, self.properties_df

    def validate_datasets(self) -> Dict[str, Any]:
        """Performs structural and data quality integrity checks on loaded datasets."""
        if self.clients_df is None or self.properties_df is None:
            self.load_data()

        audit = {
            "clients": {
                "shape": self.clients_df.shape,
                "columns": self.clients_df.columns.tolist(),
                "dtypes": self.clients_df.dtypes.astype(str).to_dict(),
                "missing_values": self.clients_df.isnull().sum().to_dict(),
                "duplicate_rows": int(self.clients_df.duplicated().sum()),
                "unique_clients": int(self.clients_df["client_id"].nunique())
            },
            "properties": {
                "shape": self.properties_df.shape,
                "columns": self.properties_df.columns.tolist(),
                "dtypes": self.properties_df.dtypes.astype(str).to_dict(),
                "missing_values": self.properties_df.isnull().sum().to_dict(),
                "duplicate_rows": int(self.properties_df.duplicated().sum()),
                "unique_listings": int(self.properties_df["listing_id"].nunique())
            }
        }
        return audit

    def inspect_relationship(self) -> Dict[str, Any]:
        """Verifies the foreign key relationship between clients.client_id and properties.client_ref."""
        if self.clients_df is None or self.properties_df is None:
            self.load_data()

        client_ids = set(self.clients_df["client_id"].dropna().unique())
        prop_client_refs = set(self.properties_df["client_ref"].dropna().unique())

        missing_refs_count = int(self.properties_df["client_ref"].isnull().sum())
        sold_count = int((self.properties_df["listing_status"] == "Sold").sum())
        available_count = int((self.properties_df["listing_status"] == "Available").sum())

        sold_missing_ref = int(((self.properties_df["listing_status"] == "Sold") & (self.properties_df["client_ref"].isnull())).sum())
        avail_missing_ref = int(((self.properties_df["listing_status"] == "Available") & (self.properties_df["client_ref"].isnull())).sum())

        invalid_refs = prop_client_refs - client_ids
        unmatched_clients = client_ids - prop_client_refs

        return {
            "total_properties": len(self.properties_df),
            "total_clients": len(self.clients_df),
            "total_sold_properties": sold_count,
            "total_available_properties": available_count,
            "total_missing_client_refs": missing_refs_count,
            "sold_properties_missing_client_ref": sold_missing_ref,
            "available_properties_missing_client_ref": avail_missing_ref,
            "invalid_client_refs_count": len(invalid_refs),
            "clients_with_no_purchases": len(unmatched_clients)
        }

    def get_sold_and_available(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits properties dataframe into sold properties (with client references) and available inventory."""
        if self.properties_df is None:
            self.load_data()

        sold_df = self.properties_df[self.properties_df["listing_status"] == "Sold"].copy()
        available_df = self.properties_df[self.properties_df["listing_status"] == "Available"].copy()

        return sold_df, available_df
