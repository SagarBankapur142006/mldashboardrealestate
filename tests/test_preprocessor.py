import unittest
import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.utils import REFERENCE_DATE

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()
        self.clients_df, self.properties_df = self.loader.load_data()
        self.preprocessor = Preprocessor(reference_date=REFERENCE_DATE)

    def test_currency_cleaning(self):
        cleaned_props = self.preprocessor.clean_properties(self.properties_df)
        self.assertIn("sale_price_clean", cleaned_props.columns)
        self.assertEqual(cleaned_props["sale_price_clean"].dtype, np.float64)
        self.assertGreater(cleaned_props["sale_price_clean"].min(), 0)
        self.assertEqual(cleaned_props["price_per_sqft"].isnull().sum(), 0)

    def test_buyer_age_calculation(self):
        cleaned_clients = self.preprocessor.clean_clients(self.clients_df)
        self.assertIn("buyer_age", cleaned_clients.columns)
        self.assertEqual(cleaned_clients["buyer_age"].isnull().sum(), 0)
        self.assertTrue((cleaned_clients["buyer_age"] >= 18).all())

    def test_buyer_profiles_building(self):
        buyer_profiles = self.preprocessor.build_buyer_profiles(self.clients_df, self.properties_df)
        self.assertEqual(len(buyer_profiles), 2000)
        self.assertIn("total_spend", buyer_profiles.columns)
        self.assertIn("properties_bought", buyer_profiles.columns)
        self.assertTrue((buyer_profiles["properties_bought"] >= 1).all())

    def test_feature_matrix_preparation(self):
        buyer_profiles = self.preprocessor.build_buyer_profiles(self.clients_df, self.properties_df)
        X_scaled, ml_cols, X_raw = self.preprocessor.prepare_feature_matrix(buyer_profiles)
        self.assertEqual(X_scaled.shape[0], 2000)
        self.assertEqual(X_scaled.shape[1], len(ml_cols))
        self.assertFalse(np.isnan(X_scaled).any())

if __name__ == "__main__":
    unittest.main()
