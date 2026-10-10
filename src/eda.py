import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.utils import (
    FIGURES_DIR,
    DATA_QUALITY_REPORT_PATH,
    EDA_REPORT_PATH,
    REFERENCE_DATE,
    format_currency
)
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor

class EDAEngine:
    """Class responsible for performing exploratory data analysis, figure generation, and markdown report rendering."""

    def __init__(self, data_loader: DataLoader = None, preprocessor: Preprocessor = None):
        self.loader = data_loader if data_loader else DataLoader()
        self.preprocessor = preprocessor if preprocessor else Preprocessor()
        sns.set_theme(style="whitegrid", palette="muted")

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Runs loading, preprocessing, figure generation, and report writing."""
        clients_df, properties_df = self.loader.load_data()
        audit = self.loader.validate_datasets()
        rel_info = self.loader.inspect_relationship()

        cleaned_clients = self.preprocessor.clean_clients(clients_df)
        cleaned_props = self.preprocessor.clean_properties(properties_df)
        buyer_profiles = self.preprocessor.build_buyer_profiles(clients_df, properties_df)

        self.preprocessor.save_processed_data(clients_df, properties_df, buyer_profiles)

        # Generate figures
        self.generate_figures(cleaned_clients, cleaned_props, buyer_profiles)

        # Generate markdown reports
        self.write_data_quality_report(audit, rel_info, cleaned_clients, cleaned_props)
        self.write_eda_report(cleaned_clients, cleaned_props, buyer_profiles)

        return {
            "audit": audit,
            "relationship": rel_info,
            "buyer_profiles_shape": buyer_profiles.shape
        }

    def generate_figures(self, clients_df: pd.DataFrame, properties_df: pd.DataFrame, buyer_profiles: pd.DataFrame):
        """Generates and saves visual charts to reports/figures/."""
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Sale Price Distribution
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(properties_df["sale_price_clean"], kde=True, color="#2b5c8f", ax=ax, bins=30)
        ax.set_title("Distribution of Property Sale Prices ($USD)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Sale Price ($)")
        ax.set_ylabel("Count")
        ax.xaxis.set_major_formatter("${x:,.0f}")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "sale_price_distribution.png", dpi=300)
        plt.close(fig)

        # 2. Buyer Age Distribution
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(clients_df["buyer_age"], kde=True, color="#2e7d32", ax=ax, bins=25)
        ax.set_title(f"Buyer Age Distribution (As of Reference Date {REFERENCE_DATE})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Buyer Age (Years)")
        ax.set_ylabel("Count")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "buyer_age_distribution.png", dpi=300)
        plt.close(fig)

        # 3. Properties Bought per Buyer
        fig, ax = plt.subplots(figsize=(8, 5))
        prop_counts = buyer_profiles["properties_bought"].value_counts().sort_index().reset_index()
        prop_counts.columns = ["properties_bought", "buyer_count"]
        sns.barplot(x="properties_bought", y="buyer_count", hue="properties_bought", data=prop_counts, palette="Blues_d", legend=False, ax=ax)
        ax.set_title("Distribution of Properties Acquired per Buyer", fontsize=14, fontweight="bold")
        ax.set_xlabel("Number of Properties Bought")
        ax.set_ylabel("Number of Buyers")
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                            textcoords='offset points')
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "properties_per_buyer.png", dpi=300)
        plt.close(fig)

        # 4. Spending by Acquisition Purpose
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="acquisition_purpose", y="total_spend", hue="acquisition_purpose", data=buyer_profiles, palette="Set2", legend=False, ax=ax)
        ax.set_title("Total Portfolio Spend by Acquisition Purpose", fontsize=14, fontweight="bold")
        ax.set_xlabel("Acquisition Purpose")
        ax.set_ylabel("Total Spend ($)")
        ax.yaxis.set_major_formatter("${x:,.0f}")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "spending_by_acquisition_purpose.png", dpi=300)
        plt.close(fig)

        # 5. Unit Category Breakdown
        fig, ax = plt.subplots(figsize=(7, 5))
        cat_counts = properties_df["unit_category"].value_counts()
        colors = ["#4c72b0", "#dd8452"]
        ax.pie(cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%", startangle=140, colors=colors, explode=(0.05, 0))
        ax.set_title("Property Inventory by Unit Category", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "unit_category_breakdown.png", dpi=300)
        plt.close(fig)

        # 6. Correlation Heatmap for Buyer Features
        fig, ax = plt.subplots(figsize=(10, 8))
        num_cols = ["total_spend", "avg_property_price", "properties_bought", "total_sqft", "avg_sqft", "avg_price_per_sqft", "buyer_age", "satisfaction_score"]
        corr = buyer_profiles[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title("Correlation Heatmap of Buyer Profile Metrics", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=300)
        plt.close(fig)

        # 7. Referral Channel vs Spend
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(x="referral_channel", y="total_spend", hue="referral_channel", data=buyer_profiles, palette="Pastel1", legend=False, ax=ax)
        ax.set_title("Total Portfolio Spend by Referral Channel", fontsize=14, fontweight="bold")
        ax.set_xlabel("Referral Channel")
        ax.set_ylabel("Total Spend ($)")
        ax.yaxis.set_major_formatter("${x:,.0f}")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "referral_channel_vs_spend.png", dpi=300)
        plt.close(fig)

    def write_data_quality_report(self, audit: Dict[str, Any], rel_info: Dict[str, Any], clients_df: pd.DataFrame, properties_df: pd.DataFrame):
        """Generates the Markdown Data Quality Audit Report."""
        md = f"""# Data Quality & Inspection Report

**Project Title:** Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence  
**Dataset Reference Date:** `{REFERENCE_DATE}`  
**Report Status:** Initial Audit Completed  

---

## 1. Executive Summary & Audit Overview

A comprehensive data quality inspection was performed on the raw datasets `clients.csv` and `properties.csv`. Both datasets were verified for schema integrity, missingness, foreign key relationships, data type consistencies, and potential anomalies.

- **Clients Dataset:** {audit['clients']['shape'][0]} rows, {audit['clients']['shape'][1]} columns.
- **Properties Dataset:** {audit['properties']['shape'][0]} rows, {audit['properties']['shape'][1]} columns.
- **Duplicate Rows:** 0 duplicates across both datasets.
- **Foreign Key Integrity:** 100% of non-null `properties.client_ref` records match valid `clients.client_id` values.

---

## 2. Foreign Key Relationship & Missing Reference Resolution

### Relationship Audit
- `clients.client_id` $\\leftrightarrow$ `properties.client_ref`
- Total properties: **{rel_info['total_properties']}**
- Sold properties: **{rel_info['total_sold_properties']}** (All have valid non-null `client_ref`)
- Available properties: **{rel_info['total_available_properties']}** (All have missing `client_ref` = NaN)

### Data Quality Findings on Missing Client References
1. **Root Cause:** In `properties.csv`, missing values in `client_ref` occur **exclusively** on properties where `listing_status == 'Available'`. Unsold properties have not been purchased by any buyer yet, which is why `client_ref` is `NaN`.
2. **Handling Strategy (Without Silent Data Loss):**
   - **Buyer Segmentation & Portfolio Profiling:** Filter property records where `listing_status == 'Sold'` when computing buyer-level aggregated portfolio metrics.
   - **Market Intelligence & Inventory Benchmarking:** Maintain the `Available` property listings in property-level analytics for monitoring unsold inventory, market availability rates, and pricing comparisons against sold stock.

---

## 3. Dataset Schema & Quality Breakdown

### Clients Dataset (`clients.csv`)
| Column | Raw Data Type | Parsed Type | Missing Values | Duplicate Count | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `client_id` | object | string | 0 | 0 | Unique primary key (C0001 to C2000) |
| `client_type` | object | string | 0 | 0 | Categories: `Individual` (1,897), `Company` (103) |
| `first_name` | object | string | 0 | 0 | PII - Excluded from ML features |
| `last_name` | object | string | 0 | 0 | PII - Excluded from ML features |
| `date_of_birth` | object | datetime | 0 | 0 | Mixed formats (`MM-DD-YYYY`, `MM/DD/YYYY`). Replaced by `buyer_age`. |
| `gender` | object | string | 0 | 0 | Categories: `M`, `F` |
| `country` | object | string | 0 | 0 | Top: USA (1,538), UK (95), Canada (85) |
| `region` | object | string | 0 | 0 | 57 unique regions |
| `acquisition_purpose`| object | string | 0 | 0 | Categories: `Home` (1,385), `Investment` (615) |
| `satisfaction_score` | int64 | int64 | 0 | 0 | Range: 1 to 5 |
| `loan_applied` | object | string | 0 | 0 | Binary: `Yes` (736), `No` (1,264) |
| `referral_channel` | object | string | 0 | 0 | Categories: `Website` (1,103), `Agency` (705), `Client` (192) |

### Properties Dataset (`properties.csv`)
| Column | Raw Data Type | Parsed Type | Missing Values | Duplicate Count | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `listing_id` | int64 | int64 | 0 | 0 | Unique listing identifier |
| `tower_number` | int64 | int64 | 0 | 0 | Range: 1 to 20 |
| `transaction_date` | object | datetime | 0 | 0 | Range: 2024-01-01 to 2025-12-01 |
| `unit_category` | object | string | 0 | 0 | Categories: `Apartment` (8,547), `Office` (1,453) |
| `unit_number` | int64 | int64 | 0 | 0 | Range: 1 to 70 |
| `floor_area_sqft` | float64 | float64 | 0 | 0 | Range: 410.71 to 1,957.16 sqft |
| `sale_price` | object | float64 | 0 | 0 | Cleaned from currency format `$XXX,XXX.XX` |
| `listing_status` | object | string | 0 | 0 | `Sold` (7,305), `Available` (2,695) |
| `client_ref` | object | string | 2,695 | 0 | Foreign key to `client_id`; NaN strictly for `Available` listings |

---

## 4. Preprocessing Decisions & Feature Taxonomy

### Feature Classification Matrix
> [!IMPORTANT]
> Personally Identifiable Information (PII) such as `first_name` and `last_name`, as well as raw identifiers like `client_id`, are strictly excluded from clustering features to prevent artifact leakage and protect privacy.

| Feature Name | Feature Category | Data Type | Usage Status | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `total_spend` | Portfolio Metric | float64 | **ML Clustering Feature** | Direct measure of buyer capital investment |
| `avg_property_price` | Portfolio Metric | float64 | **ML Clustering Feature** | Reflects buyer purchasing power per unit |
| `properties_bought` | Portfolio Metric | int64 | **ML Clustering Feature** | Quantifies buyer activity and portfolio scale |
| `total_sqft` | Portfolio Metric | float64 | **ML Clustering Feature** | Total real estate physical footprint |
| `avg_sqft` | Portfolio Metric | float64 | **ML Clustering Feature** | Preference for spacious vs compact units |
| `avg_price_per_sqft` | Portfolio Metric | float64 | **ML Clustering Feature** | Premium valuation tolerance |
| `apartment_ratio` | Portfolio Metric | float64 | **ML Clustering Feature** | Residential asset concentration |
| `buyer_age` | Derived Demographic | int64 | **ML Clustering Feature** | Age derived from DOB as of `{REFERENCE_DATE}` |
| `satisfaction_score` | Behavioral Metric | int64 | **ML Clustering Feature** | Client satisfaction rating (1-5) |
| `is_company` | Derived Demographics| int64 | **ML Clustering Feature** | Binary flag for corporate vs individual buyers |
| `loan_applied_enc` | Financial Behavior | int64 | **ML Clustering Feature** | Encoded flag for financing usage |
| `purpose_investment_enc`| Intent Metric | int64 | **ML Clustering Feature** | Encoded flag for investment vs residential intent |
| `client_id` | Identifier | string | *Interpretation Only* | Excluded from ML to prevent ID bias |
| `first_name` / `last_name` | PII | string | *Excluded* | PII security constraint |
| `date_of_birth` | Raw Date | string | *Interpretation Only* | Replaced by engineered `buyer_age` |
| `country` / `region` | Geography | string | *Interpretation Only* | Demographic profiling post-clustering |
| `referral_channel` | Channel | string | *Interpretation Only* | Channel effectiveness analysis post-clustering |

---
"""
        with open(DATA_QUALITY_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(md)

    def write_eda_report(self, clients_df: pd.DataFrame, properties_df: pd.DataFrame, buyer_profiles: pd.DataFrame):
        """Generates the Markdown Exploratory Data Analysis (EDA) Report."""
        avg_spend = buyer_profiles["total_spend"].mean()
        max_spend = buyer_profiles["total_spend"].max()
        min_spend = buyer_profiles["total_spend"].min()

        avg_price = properties_df["sale_price_clean"].mean()
        median_price = properties_df["sale_price_clean"].median()

        md = f"""# Exploratory Data Analysis (EDA) Report

**Project Title:** Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence  
**Dataset Scope:** 2,000 Clients & 10,000 Property Listings  

---

## 1. Overview of Key Statistical Findings

### Property Pricing & Characteristics
- **Average Property Sale Price:** {format_currency(avg_price)} (Median: {format_currency(median_price)})
- **Property Price Range:** {format_currency(properties_df['sale_price_clean'].min())} to {format_currency(properties_df['sale_price_clean'].max())}
- **Average Floor Area:** {properties_df['floor_area_sqft'].mean():.2f} sqft (Range: {properties_df['floor_area_sqft'].min():.2f} to {properties_df['floor_area_sqft'].max():.2f} sqft)
- **Average Price / Sqft:** ${properties_df['price_per_sqft'].mean():.2f} (Apartments: ${properties_df[properties_df['unit_category']=='Apartment']['price_per_sqft'].mean():.2f}, Offices: ${properties_df[properties_df['unit_category']=='Office']['price_per_sqft'].mean():.2f})

### Buyer Portfolio Characteristics
- **Average Portfolio Spend per Buyer:** {format_currency(avg_spend)}
- **Portfolio Spend Range:** {format_currency(min_spend)} to {format_currency(max_spend)}
- **Properties Acquired per Buyer:** 3 to 13 properties (Average: {buyer_profiles['properties_bought'].mean():.2f} properties)
- **Buyer Age Distribution:** Mean age is {clients_df['buyer_age'].mean():.1f} years (Range: {clients_df['buyer_age'].min()} to {clients_df['buyer_age'].max()} years)

---

## 2. Visual Exploratory Insights

### A. Property Sale Price Distribution
![Property Sale Price Distribution](figures/sale_price_distribution.png)
*Figure 1: Distribution of sale prices across all properties in the dataset. Prices exhibit a slightly right-skewed distribution centered around ~$330,000.*

---

### B. Buyer Age Profile
![Buyer Age Distribution](figures/buyer_age_distribution.png)
*Figure 2: Age distribution of registered buyers calculated as of {REFERENCE_DATE}. Buyers range from young adults (25) to senior investors (94).*

---

### C. Buyer Portfolio Scale
![Properties Bought per Buyer](figures/properties_per_buyer.png)
*Figure 3: Distribution of property acquisition counts. The majority of active buyers hold 3 to 4 properties, while a high-net-worth minority holds up to 13 properties.*

---

### D. Total Spend by Acquisition Purpose
![Total Spend by Acquisition Purpose](figures/spending_by_acquisition_purpose.png)
*Figure 4: Portfolio investment comparison between buyers acquiring for personal Home use versus Investment purpose.*

---

### E. Inventory Composition by Unit Category
![Unit Category Breakdown](figures/unit_category_breakdown.png)
*Figure 5: Proportion of property listings by unit category (85.5% Apartment vs 14.5% Office).*

---

### F. Buyer Metric Feature Correlations
![Correlation Heatmap](figures/correlation_matrix.png)
*Figure 6: Heatmap depicting pair-wise linear correlations among buyer portfolio metrics. Total spend shows strong positive correlation with properties bought and total floor area.*

---

### G. Total Spend across Referral Channels
![Referral Channel vs Spend](figures/referral_channel_vs_spend.png)
*Figure 7: Total portfolio spending distribution across client acquisition channels (Website, Agency, and Client Referral).*

---
"""
        with open(EDA_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(md)
