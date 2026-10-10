import unittest
import pandas as pd
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()

    def test_load_data(self):
        clients_df, properties_df = self.loader.load_data()
        self.assertIsNotNone(clients_df)
        self.assertIsNotNone(properties_df)
        self.assertEqual(len(clients_df), 2000)
        self.assertEqual(len(properties_df), 10000)

    def test_validate_datasets(self):
        audit = self.loader.validate_datasets()
        self.assertEqual(audit["clients"]["duplicate_rows"], 0)
        self.assertEqual(audit["properties"]["duplicate_rows"], 0)
        self.assertEqual(audit["clients"]["unique_clients"], 2000)

    def test_inspect_relationship(self):
        rel_info = self.loader.inspect_relationship()
        self.assertEqual(rel_info["total_sold_properties"], 7305)
        self.assertEqual(rel_info["total_available_properties"], 2695)
        self.assertEqual(rel_info["total_missing_client_refs"], 2695)
        self.assertEqual(rel_info["sold_properties_missing_client_ref"], 0)
        self.assertEqual(rel_info["invalid_client_refs_count"], 0)

if __name__ == "__main__":
    unittest.main()
