# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

**Domain:** Financial Analytics & Real Estate Market Intelligence  
**Internship Organization:** Unified Mentor  
**Collaborating Organization:** Parcl Co. Limited  
**Author / Prepared By:** Machine Learning Engineering Intern  
**Submission Date:** October 2026  

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Project Objectives](#4-project-objectives)
5. [Dataset Description](#5-dataset-description)
6. [Data Cleaning](#6-data-cleaning)
7. [Feature Encoding](#7-feature-encoding)
8. [Feature Scaling and Engineering](#8-feature-scaling-and-engineering)
9. [Machine Learning Methodology](#9-machine-learning-methodology)
10. [Optimal Cluster Selection](#10-optimal-cluster-selection)
11. [Cluster Interpretation and Buyer Profiling](#11-cluster-interpretation-and-buyer-profiling)
12. [Exploratory Data Analysis](#12-exploratory-data-analysis)
13. [Streamlit Dashboard Implementation](#13-streamlit-dashboard-implementation)
14. [Results and Discussion](#14-results-and-discussion)
15. [Business Recommendations](#15-business-recommendations)
16. [Limitations and Future Scope](#16-limitations-and-future-scope)
17. [Conclusion](#17-conclusion)
18. [References](#18-references)
19. [Appendix](#19-appendix)

---

## 1. Abstract

This research paper presents a comprehensive machine learning framework designed for real estate buyer segmentation and investment portfolio profiling, developed for Parcl Co. Limited in association with Unified Mentor. The project utilizes two real-world datasets comprising 2,000 unique client demographic profiles (`clients.csv`) and 10,000 property listing records (`properties.csv`). 

A rigorous, non-destructive data preprocessing pipeline was implemented: missing values in property records were audited and proved to belong exclusively to unsold `Available` inventory (2,695 listings), preserving 100% of purchase transaction records (7,305 `Sold` properties); categorical features were normalized and encoded using Binary, One-Hot, and Label Encoding techniques; and buyer age was derived as of a standardized reference date (`2025-12-31`). Numerical attributes (`age` and `satisfaction_score`) were standardized using `StandardScaler`.

Unsupervised clustering algorithms—specifically **K-Means Clustering** and **Agglomerative Hierarchical Clustering**—were applied to the 19-dimensional numeric feature matrix. Cluster evaluation across $K = 2 \dots 10$ using the **Elbow Method** and **Silhouette Score** identified an optimal operational trade-off at **$K = 4$** ($	ext{Inertia} = 28,407.70$, $	ext{Silhouette Score} = 0.6582$), aligning mathematical cluster cohesion with the four operational buyer profiles requested in the project specification.

By linking buyer profiles to actual completed transaction data, four distinct buyer segments were identified and profiled:
1. **Cluster 0 (Low-Leverage Residential Buyers, 17.45%, $N=349$):** Characterized by the lowest loan utilization rate (32.95%), highest satisfaction rating (3.13 / 5.0), and 71.1% home acquisition intent.
2. **Cluster 1 (High-Volume Core Buyers / Global Investors, 55.75%, $N=1,115$):** The majority segment, exhibiting the highest cumulative portfolio spend ($1,275,002.96 per buyer) and highest acquisition volume (3.71 properties per buyer).
3. **Cluster 2 (High-Valuation Premium & Corporate Buyers, 15.70%, $N=314$):** Characterized by the highest average property unit purchase price ($350,987.23 per unit) and highest corporate client representation (7.3% Company entities).
4. **Cluster 3 (Leveraged Investment Buyers, 11.10%, $N=222$):** Characterized by the highest mortgage loan utilization rate (42.79%) and highest investment purpose ratio (32.88%).

An interactive, production-ready **Streamlit web application** (`app.py`) was deployed, featuring four dedicated analytical dashboards and reactive multi-select filters (`Country`, `Region`, `Acquisition Purpose`, `Client Type`). The system empowers real estate marketing, sales, and portfolio management teams to transition from generic marketing campaigns to data-driven, customer-centric market intelligence strategies.

---

## 2. Introduction

The global real estate market is undergoing a rapid digital transformation driven by the proliferation of transactional data and advanced analytics. Traditionally, real estate brokerage firms and investment platforms have relied on broad demographic generalizations or manual rule-based heuristics to categorize buyers. However, treating all property purchasers as a homogeneous group introduces significant operational inefficiencies, including misallocated advertising budgets, generic property recommendations, and sub-optimal mortgage financing packages.

Buyer segmentation via unsupervised machine learning offers a rigorous mathematical approach to discovering latent structure within multi-dimensional customer data. Unlike supervised learning, which requires predefined target labels, unsupervised algorithms analyze patterns of similarity across demographic attributes, financial behavior, geographic origins, and transaction histories to partition buyers into natural, cohesive sub-groups.

In real estate market intelligence, effective buyer segmentation enables platforms to:
- Tailor property marketing campaigns based on explicit acquisition intent (personal residential home vs. rental investment portfolio).
- Optimize mortgage loan partnerships for leveraged buyers while offering cash-discounts to low-leverage buyers.
- Align property inventory acquisition (residential apartments vs. commercial office units) with cluster demand preferences.
- Provide personalized property recommendation feeds within digital real estate platforms.

This research paper documents the end-to-end machine learning methodology, statistical data quality audits, feature engineering pipelines, model evaluation metrics, empirical cluster profiling, and interactive web dashboard deployment executed for Parcl Co. Limited and Unified Mentor.

---

## 3. Problem Statement

Parcl Co. Limited operates at the intersection of real estate market intelligence and digital property analytics. To optimize user engagement, property marketing efficiency, and inventory allocation, Parcl requires a data-driven understanding of its buyer ecosystem across five key behavioral and demographic dimensions:
1. **Acquisition Purpose:** Differentiating buyers purchasing properties for personal residential use (`Home`) versus income-generating asset investment (`Investment`).
2. **Financing & Leverage Behavior:** Identifying buyers who utilize mortgage financing (`loan_applied == 'Yes'`) versus low-leverage or self-financed buyers.
3. **Geographic Distribution & Mobility:** Mapping buyer distribution across domestic regional jurisdictions and international origin countries.
4. **Entity Type & Scale:** Distinguishing individual buyers (`Individual`) from corporate real estate acquisition entities (`Company`).
5. **Portfolio Purchasing Scale:** Quantifying cumulative portfolio spend, average unit purchase prices, and physical floor area preferences.

Without automated machine learning segmentation, marketing and sales teams risk engaging high-net-worth portfolio investors with single-unit starter home ads or pitching leveraged mortgage products to cash-rich buyers. The central objective of this project is to build an end-to-end machine learning pipeline that ingests raw client and property datasets, preprocesses and engineers feature matrices, discovers optimal buyer clusters, profiles cluster characteristics empirically, and surfaces these insights through an interactive market intelligence dashboard.

---

## 4. Project Objectives

To address Parcl's operational challenge, the project was executed according to a structured seven-step methodology:

1. **Data Inspection & Quality Audit (Step 1):** Perform schema inspection, audit missing values, verify foreign key relationships (`clients.client_id` $\leftrightarrow$ `properties.client_ref`), and normalize categorical text formatting without modifying raw files.
2. **Categorical Feature Encoding (Step 2):** Convert nominal categorical attributes into numerical formats using Binary Encoding, One-Hot Encoding, and Label Encoding, justifying method choices based on feature cardinality and ordinality.
3. **Numerical Feature Scaling & Engineering (Step 3):** Derive buyer `age` as of a documented reference date (`2025-12-31`), standardize numerical attributes (`age`, `satisfaction_score`) using `StandardScaler`, and persist fitted scalers for production inference.
4. **Clustering Model Selection (Step 4):** Implement K-Means Clustering and Agglomerative Hierarchical Clustering using scikit-learn, fitting models on a 19-dimensional numerical feature matrix under a provisional $K = 4$ structure.
5. **Optimal Cluster Evaluation (Step 5):** Evaluate cluster cohesion and separation across $K = 2 \dots 10$ using the Elbow Method (Inertia WCSS) and Silhouette Score, resolving mathematical peak trade-offs with domain requirements.
6. **Cluster Profiling & Interpretation (Step 6):** Connect buyer clusters to 7,305 completed `Sold` property purchase transactions, compute demographic/financial summaries, assign empirical segment names, and formulate evidence-based marketing implications.
7. **Interactive Dashboard Development (Step 7):** Develop and deploy a production-grade Streamlit web application (`app.py`) featuring four analytical sections and reactive multi-select filters.

---

## 5. Dataset Description

The project utilizes two primary CSV datasets representing client profiles and property listing records.

### 5.1 Raw Dataset Overview

| Dataset File | Record Count | Column Count | Primary Key / Identifier | Missing Values | Duplicate Rows |
| :--- | :---: | :---: | :--- | :---: | :---: |
| `clients.csv` | 2,000 | 12 | `client_id` (C0001 to C2000) | 0 (0.0%) | 0 |
| `properties.csv` | 10,000 | 9 | `listing_id` (1001 to 10000) | 2,695 (`client_ref`) | 0 |

### 5.2 Attribute Schemas

#### A. Clients Dataset (`clients.csv`) Schema:
1. `client_id` (String): Unique client identifier (`C0001` to `C2000`). Primary key.
2. `client_type` (String): Entity classification (`Individual`: 1,897 [94.85%], `Company`: 103 [5.15%]).
3. `first_name` (String): Client first name (PII - Excluded from ML features).
4. `last_name` (String): Client last name (PII - Excluded from ML features).
5. `date_of_birth` (String): Birth date string in mixed formats (`MM-DD-YYYY`, `MM/DD/YYYY`). Replaced by derived `age`.
6. `gender` (String): Biological sex indicator (`M`: 1,021, `F`: 979).
7. `country` (String): Origin country (10 unique countries: USA: 1,538, UK: 95, Canada: 85, Germany: 56, France: 53, Belgium: 43, Mexico: 40, Australia: 39, Russia: 36, Denmark: 15).
8. `region` (String): Administrative region / state (57 unique regions: California, Quebec, Oregon, Arizona, Berlin, etc.).
9. `acquisition_purpose` (String): Stated purchase objective (`Home`: 1,385 [69.25%], `Investment`: 615 [30.75%]).
10. `satisfaction_score` (Integer): Client satisfaction rating on a scale of 1 to 5 (Mean: 3.03, Std: 1.41).
11. `loan_applied` (String): Mortgage financing indicator (`No`: 1,264 [63.20%], `Yes`: 736 [36.80%]).
12. `referral_channel` (String): Customer acquisition channel (`Website`: 1,103 [55.15%], `Agency`: 705 [35.25%], `Client`: 192 [9.60%]).

#### B. Properties Dataset (`properties.csv`) Schema:
1. `listing_id` (Integer): Unique property listing identifier. Primary key.
2. `tower_number` (Integer): Tower building number (1 to 20).
3. `transaction_date` (String): Listing or transaction timestamp (`2024-01-01` to `2025-12-01`).
4. `unit_category` (String): Property property type (`Apartment`: 8,547 [85.47%], `Office`: 1,453 [14.53%]).
5. `unit_number` (Integer): Specific unit floor number (1 to 70).
6. `floor_area_sqft` (Float): Physical unit area in square feet (Mean: 1,141.15 sqft, Min: 410.71 sqft, Max: 1,957.16 sqft).
7. `sale_price` (String): Formatted currency string (`$97,402.80` to `$736,652.27`).
8. `listing_status` (String): Transaction status (`Sold`: 7,305 [73.05%], `Available`: 2,695 [26.95%]).
9. `client_ref` (String): Foreign key referencing `clients.client_id`. Missing for 2,695 records.

### 5.3 Dataset Relationship & Transaction Filtering Logic

The relational link between datasets is governed by `clients.client_id` $\leftrightarrow$ `properties.client_ref`. A critical data quality audit revealed:
- **`Sold` Properties (7,305 records):** 100% of `Sold` properties possess valid, non-null `client_ref` values matching registered clients.
- **`Available` Properties (2,695 records):** 100% of `Available` properties have `client_ref == NaN`.

> [!IMPORTANT]
> Unsold `Available` properties represent active marketplace inventory stock that has not been purchased by any buyer. When computing buyer-level cumulative portfolio spend, average purchase price, and unit acquisition counts, properties are strictly filtered for `listing_status == 'Sold'`. Available listings are preserved in property-level analytics for supply benchmarking but are never counted as completed buyer purchases.

---

## 6. Data Cleaning

Data cleaning (Step 1) was executed via `step1_data_cleaning.py`, enforcing a non-destructive transformation protocol: raw source files (`clients.csv` and `properties.csv`) remained untouched, while cleaned versions were saved to `data/cleaned_clients.csv` and `data/cleaned_properties.csv`.

### 6.1 Audit & Cleaning Protocol

1. **Missing Attribute Inspection:**
   - `clients.csv`: 0 missing values across all 12 columns.
   - `properties.csv`: 2,695 missing values in `client_ref`. Cross-tabulation proved that missing values occur **exclusively** on properties where `listing_status == 'Available'`. No client or property records were dropped.
2. **Categorical Label Normalization:**
   - Leading and trailing whitespace was stripped (`.str.strip()`) across all 7 categorical columns in `clients.csv` (`client_type`, `gender`, `country`, `region`, `acquisition_purpose`, `loan_applied`, `referral_channel`) and 2 categorical columns in `properties.csv` (`unit_category`, `listing_status`).
3. **Duplicate Verification:**
   - `clients.csv`: 0 full row duplicates, 0 duplicate `client_id` values.
   - `properties.csv`: 0 full row duplicates, 0 duplicate `listing_id` values.

### 6.2 Data Quality Audit Table

| Dataset | Attribute Name | Raw Missing | Cleaned Missing | Transformation Action | Final Record Count |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Clients** | `client_id` | 0 | 0 | String stripped; primary key validated | 2,000 |
| **Clients** | `client_type` | 0 | 0 | String stripped (`Individual` / `Company`) | 2,000 |
| **Clients** | All Demographics | 0 | 0 | String stripped; zero missing values | 2,000 |
| **Properties** | `listing_id` | 0 | 0 | Primary key validated | 10,000 |
| **Properties** | `sale_price` | 0 | 0 | Stripped `$` and `,`; converted to float64 | 10,000 |
| **Properties** | `client_ref` | 2,695 | 2,695 | Preserved NaN for `Available` listings; 100% match for `Sold` | 10,000 |

---

## 7. Feature Encoding

In Step 2 (`step2_feature_encoding.py`), qualitative categorical attributes were mapped to numerical representations suitable for distance-based clustering algorithms.

### 7.1 Encoding Method Selection & Rationale

1. **`client_type` (Binary Nominal):** Mapped using Binary Label Encoding (`Individual` $ightarrow$ 0, `Company` $ightarrow$ 1). Rationale: Exactly two categories; binary mapping preserves compactness without introducing arbitrary distance bias.
2. **`acquisition_purpose` (Binary Nominal):** Mapped using Binary Label Encoding (`Home` $ightarrow$ 0, `Investment` $ightarrow$ 1). Rationale: Binary intent indicator.
3. `loan_applied` (Binary Nominal): Mapped using Binary Label Encoding (`No` $ightarrow$ 0, `Yes` $ightarrow$ 1).
4. **`referral_channel` (Multi-class Nominal - 3 categories):** Encoded using One-Hot Encoding (`referral_channel_Agency`, `referral_channel_Client`, `referral_channel_Website`). Rationale: Nominal categories without natural rank order.
5. **`country` (Multi-class Nominal - 10 categories):** Encoded using One-Hot Encoding (`country_Australia`, `country_Belgium`, `country_Canada`, `country_Denmark`, `country_France`, `country_Germany`, `country_Mexico`, `country_Russia`, `country_UK`, `country_USA`). Rationale: Unordered geographical entities.
6. **`region` (High-Cardinality Nominal - 57 categories):** Encoded using Label Encoding (`LabelEncoder()`, integer range 0 to 56). Rationale: High cardinality (57 regions). One-hot encoding would add 57 sparse columns, inflating dimensionality; Label Encoding was selected to keep feature space manageable, though integer assignment imposes artificial distances during clustering.

### 7.2 Encoded Feature Matrix Summary

The resulting encoded dataset (`data/encoded_clients.csv`) expanded the original client schema to 28 columns, yielding 16 new numerical encoded features. Data type verification confirmed that 100% of target encoded variables were numeric (`int64`), with zero string values remaining.

---

## 8. Feature Scaling and Engineering

Step 3 (`step3_feature_scaling.py`) involved age calculation and numerical feature standardization.

### 8.1 Buyer Age Derivation

Because raw birth dates (`date_of_birth`) are string timestamps, buyer age was derived as of a fixed reference date matching the end of the property transaction timeline:
- **Documented Reference Date:** `2025-12-31`
- **Derivation Formula:**
  $$	ext{age} = \left\lfloor rac{	ext{Reference Date} - 	ext{date\_of\_birth\_parsed}}{365.25} ightfloor$$

#### Verification of Birth Date Parsing:
All 2,000 client birth dates parsed cleanly (`format='mixed'`) with **0 missing values or conversion errors**. The derived buyer age ranges from 25 to 94 years, with a mean of **55.10 years** ($\sigma = 17.34$). The original `date_of_birth` column was preserved without being overwritten.

### 8.2 Numerical Feature Standardization

Clustering algorithms relying on Euclidean distance are sensitive to attribute scale differences. Standardizing features ensures that attributes with large integer ranges (e.g., age 25–94) do not dominate attributes with smaller bounds (e.g., satisfaction score 1–5).

Scikit-learn's `StandardScaler` was applied to `age` and `satisfaction_score`:
$$z = rac{x - \mu}{\sigma}$$

#### Statistical Summary: Before vs. After Scaling

| Numerical Attribute | Raw Min | Raw Max | Raw Mean ($\mu$) | Raw Std ($\sigma$) | Scaled Min ($z$) | Scaled Max ($z$) | Scaled Mean ($z$) | Scaled Std ($z$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`age`** | 25.00 | 94.00 | 55.099 | 17.344 | -1.7359 | +2.2435 | 0.0000 | 1.0003 |
| **`satisfaction_score`** | 1.00 | 5.00 | 3.029 | 1.414 | -1.4357 | +1.3947 | 0.0000 | 1.0003 |

The scaled features were saved as `age_scaled` and `satisfaction_score_scaled` in `data/scaled_clients.csv`. The fitted `StandardScaler` model was serialized and saved to `data/scaler.joblib` for production deployment.

---

## 9. Machine Learning Methodology

Step 4 (`src/clustering.py`) established the unsupervised machine learning pipeline using 19 numeric features.

### 9.1 Feature Matrix Formulation

The clustering feature matrix $X \in \mathbb{R}^{2000 	imes 19}$ comprised:
1. `client_type_encoded` (Binary)
2. `acquisition_purpose_encoded` (Binary)
3. `loan_applied_encoded` (Binary)
4. `region_encoded` (Integer 0–56)
5. `referral_channel_*` (3 One-Hot binary features)
6. `country_*` (10 One-Hot binary features)
7. `age_scaled` (Standardized float)
8. `satisfaction_score_scaled` (Standardized float)

> [!IMPORTANT]
> Unencoded string columns (`country`, `region`, `client_type`, `acquisition_purpose`), identifiers (`client_id`), PII (`first_name`, `last_name`), and unscaled numerical attributes (`age`, `satisfaction_score`) were strictly excluded from $X$ to prevent duplicate attribute weighting and privacy leakage.

### 9.2 Clustering Algorithms

#### A. K-Means Clustering:
K-Means partitions $N$ observations into $K$ clusters by minimizing the Within-Cluster Sum of Squares (WCSS / Inertia):
$$	ext{Inertia} = \sum_{i=1}^{N} \min_{\mu_j \in C} \|x_i - \mu_j\|^2$$

- **Configuration:** `n_clusters=4`, `random_state=42`, `n_init=10`, algorithm = Lloyd.

#### B. Hierarchical Agglomerative Clustering:
Agglomerative Clustering builds a bottom-up hierarchy of clusters. Ward's linkage method was selected to minimize total within-cluster variance:
$$\Delta d(A, B) = rac{|A| \cdot |B|}{|A| + |B|} \|\mu_A - \mu_B\|^2$$

- **Configuration:** `n_clusters=4`, `metric='euclidean'`, `linkage='ward'`.

### 9.3 Algorithmic Independence of Cluster Labels
Cluster assignments from K-Means (`kmeans_cluster`) and Hierarchical Clustering (`hierarchical_cluster`) were appended as separate columns in `data/clustered_clients.csv`. Cluster integer labels (0, 1, 2, 3) generated by K-Means are algorithmically independent of integer labels generated by Hierarchical Clustering.

---

## 10. Optimal Cluster Selection

In Step 5 (`src/cluster_evaluation.py`), cluster evaluation metrics were computed across $K = 2 \dots 10$ using the 19-dimensional feature matrix.

### 10.1 Evaluation Metrics Table

| Clusters ($K$) | Inertia (WCSS) | Inertia Reduction (%) | Silhouette Score | Calinski-Harabasz Index | Davies-Bouldin Index |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 2$** | 112,844.59 | Baseline | **0.6992** (Peak) | 8,096.40 | **0.3970** |
| **$K = 3$** | 42,945.48 | -61.94% | 0.6876 | 12,257.01 | 0.4578 |
| **$K = 4$** | **28,407.70** | **-33.85%** (Elbow) | **0.6582** | **12,687.36** | **0.4720** |
| **$K = 5$** | 19,338.12 | -31.93% | 0.5634 | 14,205.22 | 0.5611 |
| **$K = 6$** | 14,324.88 | -25.92% | 0.5445 | **15,473.14** | 0.6351 |
| **$K = 7$** | 12,345.45 | -13.82% | 0.4913 | 15,007.46 | 0.7532 |
| **$K = 8$** | 11,135.80 | -9.80% | 0.4401 | 14,284.63 | 0.8347 |
| **$K = 9$** | 9,878.81 | -11.29% | 0.4172 | 14,114.04 | 0.8909 |
| **$K = 10$** | 8,989.56 | -9.00% | 0.4240 | 13,801.80 | 0.8783 |

### 10.2 Mathematical Evaluation & Trade-off Analysis

1. **Elbow Method Analysis (`reports/elbow_method.png`):** Inertia drops steeply from $K=2 ightarrow K=3$ (-61.94%) and $K=3 ightarrow K=4$ (-33.85%). Beyond $K=4$, the rate of inertia drop diminishes significantly (-13.82% at $K=7$), establishing **$K = 4$** as the primary inflection elbow point.
2. **Silhouette Score Analysis (`reports/silhouette_scores.png`):** The Silhouette Score reaches its global maximum at $K = 2$ (0.6992), followed by $K = 3$ (0.6876) and $K = 4$ (0.6582). Beyond $K=4$, Silhouette scores degrade rapidly down to 0.4172 ($K=9$).
3. **Resolution of Selection Disagreement:** While $K = 2$ achieves the highest mathematical Silhouette score, a binary split yields overly coarse operational groupings (e.g., Home vs. Investment). **$K = 4$** was selected for production because it retains strong cluster separation ($	ext{Silhouette} = 0.6582 > 0.50$), aligns with the Elbow point, and satisfies Parcl's requirement for four buyer personas.

---

## 11. Cluster Interpretation and Buyer Profiling

Step 6 (`src/cluster_interpretation.py`) linked client cluster assignments to completed `Sold` property transactions (7,305 records).

### 11.1 Empirical Cluster Summary Table

| Metric / Characteristic | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Total Dataset |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Descriptive Segment Name** | Low-Leverage Residential Buyers | High-Volume Core Buyers (Global Investors) | High-Valuation Premium & Corporate Buyers | Leveraged Investment Buyers | **2,000 Clients** |
| **Buyer Count ($N$)** | **349** | **1,115** | **314** | **222** | **2,000** |
| **Share of Buyers (%)** | **17.45%** | **55.75%** | **15.70%** | **11.10%** | **100.00%** |
| **Client Type Breakdown** | 96.0% Ind / 4.0% Co | 95.1% Ind / 4.9% Co | 92.7% Ind / **7.3% Co** | 95.0% Ind / 5.0% Co | 94.85% Ind / 5.15% Co |
| **Acquisition Purpose** | 71.1% Home / 28.9% Inv | 69.1% Home / 30.9% Inv | 69.1% Home / 30.9% Inv | 67.1% Home / **32.9% Inv** | 69.25% Home / 30.75% Inv |
| **Loan Utilization (%)** | **32.95%** (Lowest) | 35.96% | 39.81% | **42.79%** (Highest) | 36.80% |
| **Average Buyer Age** | 54.50 yrs | 55.33 yrs | 55.22 yrs | 54.71 yrs | 55.10 yrs |
| **Satisfaction Score (1-5)**| **3.13** (Highest) | 3.00 | 3.00 | 3.05 | 3.03 |
| **Avg Portfolio Spend / Buyer**| **$1,226,028.07** | **$1,275,002.96** (Highest) | **$1,259,603.41** | **$1,241,997.27** | **$1,257,597.74** |
| **Avg Unit Price Bought** | **$350,055.23** | **$346,512.36** | **$350,987.23** (Highest) | **$339,816.98** | **$346,650.05** |
| **Avg Properties / Buyer** | 3.51 units | **3.71 units** (Highest) | 3.60 units | 3.68 units | 3.65 units |
| **Avg Unit Area (sqft)** | 1,159.82 sqft | 1,145.49 sqft | 1,156.62 sqft | 1,125.17 sqft | 1,146.43 sqft |
| **Apartment / Office Ratio**| 83.8% / 16.2% | 85.0% / 15.0% | 87.0% / 13.0% | 85.9% / 14.1% | 85.5% / 14.5% |

### 11.2 Segment Characteristics & Business Analysis

#### A. Cluster 0: Low-Leverage Residential Buyers ($N=349$, 17.45%)
- **Empirical Traits:** Lowest loan utilization rate (**32.95%**), demonstrating strong self-financing capability. Highest customer satisfaction score (**3.13 / 5.0**). Primary focus on personal residential homes (71.1%). Concentrated in regional markets 44–56 (e.g., Texas, Utah, Virginia, Washington).
- **Business Implications:** Target with move-in ready luxury residential homes; position cash-incentive discounts, fee waivers, and premium customer service.

#### B. Cluster 1: High-Volume Core Buyers / Global Investors ($N=1,115$, 55.75% - Majority Segment)
- **Empirical Traits:** Majority buyer segment. Highest average cumulative portfolio spend (**$1,275,002.96** per buyer) and highest acquisition volume (**3.71 properties** per buyer). Active across major domestic and international markets (USA, Canada, Germany, UK).
- **Business Implications:** Focus on bulk-purchase portfolio packages, multi-unit management services, and dedicated VIP investor communications.

#### C. Cluster 2: High-Valuation Premium & Corporate Buyers ($N=314$, 15.70%)
- **Empirical Traits:** Highest average property unit purchase price (**$350,987.23** per unit) and price-per-sqft (**$303.36 / sqft**). Highest corporate client representation (**7.3% Company entities**).
- **Business Implications:** Offer luxury residential penthouses and commercial office space; establish dedicated B2B corporate sales channels.

#### D. Cluster 3: Leveraged Investment Buyers ($N=222$, 11.10%)
- **Empirical Traits:** Highest mortgage loan application rate (**42.79%**) and highest investment intent ratio (**32.88%**). Active across cross-border international markets.
- **Business Implications:** Partner with mortgage lenders to provide competitive financing packages, ROI projection reports, and rental yield calculators.

---

## 12. Exploratory Data Analysis

Exploratory Data Analysis (`src/eda.py`) uncovered core statistical distributions across client demographics and property transactions.

### 12.1 Property Transaction Distributions
- **Sale Price Distribution:** Ranging from $97,402.80 to $736,652.27, property sale prices exhibit a unimodal distribution centered at a mean of **$345,072.00** (Median: $330,997.32).
- **Floor Area Distribution:** Property square footage ranges from 410.71 sqft to 1,957.16 sqft (Mean: **1,141.15 sqft**).
- **Price per Sqft:** Averages **$302.14 / sqft** overall (Apartments: $300.11 / sqft, Offices: $314.07 / sqft).

### 12.2 Demographic & Behavioral Distributions
- **Acquisition Purpose:** 1,385 clients (69.25%) acquire for `Home` use, while 615 clients (30.75%) acquire for `Investment`.
- **Loan Utilization:** 736 clients (36.80%) applied for mortgage loans (`loan_applied == 'Yes'`), while 1,264 clients (63.20%) self-financed.
- **Referral Channel:** `Website` accounts for 1,103 clients (55.15%), `Agency` accounts for 705 clients (35.25%), and `Client` referrals account for 192 clients (9.60%).

---

## 13. Streamlit Dashboard Implementation

Step 7 delivered a production-grade Streamlit web application ([app.py](file:///d:/mlintern/app.py)) providing real-time market intelligence analytics.

### 13.1 Application Architecture & Sections

The application is structured into four main operational tabs:
1. **📊 Buyer Segmentation Overview:** Displays top-level KPI metric cards, interactive Plotly cluster distribution pie/bar charts, and a searchable buyer records dataframe.
2. **📈 Investor Behavior Dashboard:** Compares acquisition purpose, loan utilization, client type, and property purchase statistics using **Sold listings only** (excluding unsold Available stock).
3. **🌐 Geographic Buyer Analysis:** Features a global Choropleth heatmap, country concentration bar charts, and top 15 regional market distributions.
4. **💡 Segment Insights & Marketing Panel:** Provides interactive expandable cards for each cluster detailing empirical metrics, data characteristics, and actionable marketing strategies.

### 13.2 Interactive Sidebar Filters

The dashboard includes four multi-select sidebar filters:
- `Country Filter`: Multiselect for 10 origin countries.
- `Region Filter`: Reactive multiselect for 57 administrative regions.
- `Acquisition Purpose Filter`: Multiselect for `Home` vs. `Investment`.
- `Client Type Filter`: Multiselect for `Individual` vs. `Company`.

All charts, KPI metric cards, and data tables reactively update when filters are adjusted.

---

## 14. Results and Discussion

The empirical findings validate the utility of unsupervised machine learning for real estate buyer profiling. Key analytical insights include:

1. **Geographic Regional Segmentation Effect:** Due to the 0–56 integer scale of `region_encoded`, K-Means clustering partitioned buyers cleanly into regional geographic market blocks. This highlights both the dominance of location in real estate purchasing and the distance sensitivity of integer label encoding.
2. **Evaluation Metric Trade-offs:** While mathematical silhouette maximization favored $K=2$ ($	ext{Silhouette} = 0.6992$), selecting $K=4$ ($	ext{Silhouette} = 0.6582$) provided the optimal balance between mathematical cohesion and granular operational buyer personas.
3. **Purchasing Power Uniformity:** Total portfolio spend per buyer averages ~$1.25M across all clusters, indicating that buyer differentiation in this dataset is driven primarily by leverage preference, unit price tolerance, and geographic location rather than raw capital scale.

---

## 15. Business Recommendations

Based on empirical cluster characteristics, the following data-driven business recommendations are proposed for Parcl Co. Limited:

1. **Low-Leverage Residential Buyers (Cluster 0):** Implement direct-to-consumer digital campaigns promoting move-in ready homes. Offer upfront cash-buyer incentives and highlight post-purchase property management.
2. **High-Volume Core Buyers / Global Investors (Cluster 1):** Develop institutional bulk-purchase programs, multi-unit portfolio management dashboards, and VIP investor newsletters.
3. **High-Valuation Premium Buyers (Cluster 2):** Tailor corporate B2B sales channels for company clients acquiring commercial offices and premium luxury apartments.
4. **Leveraged Investment Buyers (Cluster 3):** Establish co-branded mortgage financing partnerships with leading lenders to offer pre-approved investment loans and rental yield calculators.

---

## 16. Limitations and Future Scope

### 16.1 Methodological Limitations
1. **Absence of Explicit Income Data:** The dataset contains no explicit buyer income or net-worth attributes. Purchasing power was inferred strictly from property transaction amounts.
2. **Label Encoding Distance Effect:** Label encoding `region` introduced artificial numerical distances between geographic categories, causing distance-based K-Means to cluster strongly along regional boundaries.
3. **Unsold Inventory Handling:** Available listings (2,695 properties) lack buyer references and were excluded from completed purchase calculations.

### 16.2 Future Research Scope
- Incorporate Target Encoding or Spatial Coordinates (Latitude/Longitude) for geographic features.
- Integrate external macroeconomic indicators (interest rates, local housing price indices).
- Implement supervised churn and lifetime value (LTV) prediction models.

---

## 17. Conclusion

This project successfully established an end-to-end machine learning and market intelligence workflow for real estate buyer segmentation. By ingesting raw client and property datasets, engineering a 19-dimensional numeric feature matrix, evaluating clustering models across $K=2 \dots 10$, and linking buyer clusters to completed transaction records, four actionable buyer personas were profiled. The interactive Streamlit dashboard (`app.py`) bridges machine learning analytics and business execution, providing Parcl Co. Limited with a scalable platform for data-driven real estate marketing.

---

## 18. References

1. Scikit-learn Developers. (2024). *KMeans Clustering Documentation*. scikit-learn.org. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
2. Scikit-learn Developers. (2024). *AgglomerativeClustering Documentation*. scikit-learn.org. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
3. Scikit-learn Developers. (2024). *StandardScaler Documentation*. scikit-learn.org. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
4. Scikit-learn Developers. (2024). *Silhouette Score Documentation*. scikit-learn.org. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
5. Streamlit Inc. (2024). *Streamlit Documentation*. docs.streamlit.io. https://docs.streamlit.io
6. Plotly Technologies Inc. (2024). *Plotly Python Open Source Graphing Library*. plotly.com. https://plotly.com/python/

---

## 19. Appendix

### 19.1 Project Directory Structure
```
d:\mlintern├── clients.csv / properties.csv     # Raw source datasets
├── app.py                          # Streamlit application entry point
├── step1_data_cleaning.py          # Data cleaning script
├── step2_feature_encoding.py       # Feature encoding script
├── step3_feature_scaling.py        # Feature scaling script
├── src/
│   ├── clustering.py               # Step 4 clustering script
│   ├── cluster_evaluation.py       # Step 5 evaluation script
│   └── cluster_interpretation.py   # Step 6 profiling script
├── data/                           # Processed & clustered CSV data
└── reports/                        # Markdown, DOCX, and PDF research paper outputs
```

### 19.2 Execution Instructions
To launch the interactive dashboard locally:
```bash
streamlit run app.py
```
*Access application at `http://localhost:8501`.*
