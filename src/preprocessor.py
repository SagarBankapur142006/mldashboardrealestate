import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from src.utils import REFERENCE_DATE, CLEANED_CLIENTS_PATH, CLEANED_PROPERTIES_PATH, BUYER_PROFILES_PATH

class Preprocessor:
    """Class responsible for data cleaning, date parsing, currency conversion, feature engineering, and feature matrix preparation."""

    def __init__(self, reference_date: str = REFERENCE_DATE):
        self.reference_date = pd.to_datetime(reference_date)
        self.scaler = StandardScaler()

    def clean_clients(self, clients_df: pd.DataFrame) -> pd.DataFrame:
        """Cleans clients dataset, parses date_of_birth, and calculates buyer_age."""
        df = clients_df.copy()

        # Parse date_of_birth safely
        df["date_of_birth_parsed"] = pd.to_datetime(df["date_of_birth"], format="mixed", errors="coerce")

        # Calculate buyer age as of reference date
        age_in_days = (self.reference_date - df["date_of_birth_parsed"]).dt.days
        df["buyer_age"] = (age_in_days / 365.25).apply(np.floor).astype(int)

        # Encode categorical flags useful for modeling
        df["is_company"] = (df["client_type"] == "Company").astype(int)
        df["loan_applied_enc"] = (df["loan_applied"] == "Yes").astype(int)
        df["purpose_investment_enc"] = (df["acquisition_purpose"] == "Investment").astype(int)

        return df

    def clean_properties(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """Cleans properties dataset, parses transaction_date, cleans sale_price, and derives price_per_sqft."""
        df = properties_df.copy()

        # Parse transaction_date
        df["transaction_date_parsed"] = pd.to_datetime(df["transaction_date"], format="mixed", errors="coerce")
        df["transaction_year"] = df["transaction_date_parsed"].dt.year
        df["transaction_month"] = df["transaction_date_parsed"].dt.month

        # Clean sale_price currency string -> numeric float
        df["sale_price_clean"] = (
            df["sale_price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

        # Derived property metric
        df["price_per_sqft"] = df["sale_price_clean"] / df["floor_area_sqft"]

        return df

    def build_buyer_profiles(self, clients_df: pd.DataFrame, properties_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates sold property transactions at buyer level and merges with client demographic data."""
        cleaned_clients = self.clean_clients(clients_df)
        cleaned_props = self.clean_properties(properties_df)

        # Filter strictly for sold properties to construct buyer portfolio metrics
        sold_props = cleaned_props[cleaned_props["listing_status"] == "Sold"].copy()

        # Aggregate transactions per client
        portfolio = sold_props.groupby("client_ref").agg(
            total_spend=("sale_price_clean", "sum"),
            avg_property_price=("sale_price_clean", "mean"),
            properties_bought=("listing_id", "count"),
            total_sqft=("floor_area_sqft", "sum"),
            avg_sqft=("floor_area_sqft", "mean"),
            avg_price_per_sqft=("price_per_sqft", "mean"),
            apartment_count=("unit_category", lambda x: (x == "Apartment").sum()),
            office_count=("unit_category", lambda x: (x == "Office").sum())
        ).reset_index()

        portfolio["apartment_ratio"] = portfolio["apartment_count"] / portfolio["properties_bought"]
        portfolio["office_ratio"] = portfolio["office_count"] / portfolio["properties_bought"]

        # Merge with clients dataset
        buyer_profiles = pd.merge(
            cleaned_clients,
            portfolio,
            left_on="client_id",
            right_on="client_ref",
            how="left"
        )

        # Fill any client with 0 properties (though audit showed 100% match)
        buyer_profiles["properties_bought"] = buyer_profiles["properties_bought"].fillna(0).astype(int)
        buyer_profiles["total_spend"] = buyer_profiles["total_spend"].fillna(0.0)

        return buyer_profiles

    def get_feature_taxonomy(self) -> Dict[str, List[str]]:
        """Returns the clear separation of features into Machine Learning features vs Interpretation / PII features."""
        ml_features = [
            "total_spend",
            "avg_property_price",
            "properties_bought",
            "total_sqft",
            "avg_sqft",
            "avg_price_per_sqft",
            "apartment_ratio",
            "buyer_age",
            "satisfaction_score",
            "is_company",
            "loan_applied_enc",
            "purpose_investment_enc"
        ]

        interpretation_features = [
            "client_id",
            "first_name",
            "last_name",
            "client_type",
            "date_of_birth",
            "gender",
            "country",
            "region",
            "acquisition_purpose",
            "loan_applied",
            "referral_channel"
        ]

        return {
            "ml_features": ml_features,
            "interpretation_features": interpretation_features
        }

    def prepare_feature_matrix(self, buyer_profiles_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """Extracts and scales feature matrix for machine learning clustering models."""
        taxonomy = self.get_feature_taxonomy()
        ml_cols = taxonomy["ml_features"]

        X_raw = buyer_profiles_df[ml_cols].copy()
        
        # Handle potential NaNs by filling with median
        X_raw = X_raw.fillna(X_raw.median())

        X_scaled = self.scaler.fit_transform(X_raw)

        return X_scaled, ml_cols, X_raw

    def save_processed_data(self, clients_df: pd.DataFrame, properties_df: pd.DataFrame, buyer_profiles_df: pd.DataFrame):
        """Saves cleaned and engineered datasets to processed data directory."""
        cleaned_clients = self.clean_clients(clients_df)
        cleaned_props = self.clean_properties(properties_df)

        cleaned_clients.to_csv(CLEANED_CLIENTS_PATH, index=False)
        cleaned_props.to_csv(CLEANED_PROPERTIES_PATH, index=False)
        buyer_profiles_df.to_csv(BUYER_PROFILES_PATH, index=False)
