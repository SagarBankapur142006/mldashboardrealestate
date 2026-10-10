# Data Quality & Inspection Report

**Project Title:** Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence  
**Dataset Reference Date:** `2025-12-31`  
**Report Status:** Initial Audit Completed  

---

## 1. Executive Summary & Audit Overview

A comprehensive data quality inspection was performed on the raw datasets `clients.csv` and `properties.csv`. Both datasets were verified for schema integrity, missingness, foreign key relationships, data type consistencies, and potential anomalies.

- **Clients Dataset:** 2000 rows, 12 columns.
- **Properties Dataset:** 10000 rows, 9 columns.
- **Duplicate Rows:** 0 duplicates across both datasets.
- **Foreign Key Integrity:** 100% of non-null `properties.client_ref` records match valid `clients.client_id` values.

---

## 2. Foreign Key Relationship & Missing Reference Resolution

### Relationship Audit
- `clients.client_id` $\leftrightarrow$ `properties.client_ref`
- Total properties: **10000**
- Sold properties: **7305** (All have valid non-null `client_ref`)
- Available properties: **2695** (All have missing `client_ref` = NaN)

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
| `buyer_age` | Derived Demographic | int64 | **ML Clustering Feature** | Age derived from DOB as of `2025-12-31` |
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
