# Step 6: Cluster Interpretation & Buyer Profiling Report

**Project Title:** Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence  
**Input Datasets:** `data/clustered_clients.csv` & `data/cleaned_properties.csv`  
**Output Report:** `reports/cluster_interpretation.md`  

---

## 1. Executive Summary & Segmentation Strategy

Using K-Means clustering with $K = 4$, all 2,000 buyers in the dataset were partitioned into four distinct operational segments. Each cluster was analyzed by linking buyer demographic records to actual property purchase transactions (`listing_status == 'Sold'`, 7,305 total transactions).

> [!NOTE]
> Available listings (2,695 properties) represent unsold inventory stock and were excluded from buyer purchase amounts to ensure price accuracy.

---

## 2. Cluster Profiles & Empirical Comparison Table

| Metric | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Total Dataset |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Descriptive Segment Name** | Low-Leverage Residential Buyers | High-Volume Core Buyers (Global Investors) | High-Valuation Premium Buyers | Leveraged Investment Buyers | **2,000 Clients** |
| **Buyer Count ($N$)** | **349** | **1,115** | **314** | **222** | **2,000** |
| **Share of Total Buyers (%)** | 17.45% | **55.75%** | 15.70% | 11.10% | 100.00% |
| **Client Type Breakdown** | 96.0% Ind / 4.0% Co | 95.1% Ind / 4.9% Co | 92.7% Ind / **7.3% Co** | 95.0% Ind / 5.0% Co | 94.85% Ind / 5.15% Co |
| **Investment Intent (%)** | 28.94% | 30.85% | 30.89% | **32.88%** | 30.75% |
| **Loan Utilization (%)** | 32.95% (Lowest) | 35.96% | 39.81% | **42.79%** (Highest) | 36.80% |
| **Average Buyer Age** | 54.50 yrs | 55.33 yrs | 55.22 yrs | 54.71 yrs | 55.10 yrs |
| **Satisfaction Score (1-5)**| **3.13** (Highest) | 3.00 | 3.00 | 3.05 | 3.03 |
| **Avg Total Spend / Buyer** | $1,226,028.07 | **$1,275,002.96** | $1,259,603.41 | $1,241,997.27 | $1,257,597.74 |
| **Avg Unit Price Bought** | $350,055.23 | $346,512.36 | **$350,987.23** | $339,816.98 | $346,650.05 |
| **Avg Properties / Buyer** | 3.51 | **3.71** | 3.60 | 3.68 | 3.65 |
| **Avg Unit Floor Area (sqft)**| 1,159.82 | 1,145.49 | 1,156.62 | 1,125.17 | 1,146.43 |
| **Apartment / Office Ratio** | 83.8% / 16.2% | 85.0% / 15.0% | 87.0% / 13.0% | 85.9% / 14.1% | 85.5% / 14.5% |

---

## 3. Detailed Segment Profiles & Marketing Implications

### A. Cluster 0: Low-Leverage Residential Buyers
- **Size:** 349 buyers (17.45% of total dataset).
- **Key Characteristics:** 
  - Lowest loan utilization rate (**32.95%**), demonstrating strong self-financing capacity.
  - Highest client satisfaction rating (**3.13 / 5.0**).
  - 71.1% acquire properties for personal Home use.
  - Concentrated in regional markets 44-56 (e.g., Texas, Utah, Virginia, Washington).
- **Marketing Implications:** Target with premium residential move-in ready options; position cash-incentive programs and customer experience perks.

---

### B. Cluster 1: High-Volume Core Portfolio Buyers (Global Investors)
- **Size:** 1,115 buyers (**55.75%** of total dataset - Majority Segment).
- **Key Characteristics:**
  - Highest average portfolio purchase count (**3.71 properties** per buyer).
  - Highest average cumulative spend (**$1,275,002.96** per buyer).
  - Dominates core domestic and international markets (USA, Canada, Germany, UK).
  - Holds 3,512 Apartments and 620 Offices.
- **Marketing Implications:** Focus on bulk-purchase incentives, multi-unit portfolio management services, and VIP investor communications.

---

### C. Cluster 2: High-Valuation Premium & Corporate Buyers
- **Size:** 314 buyers (15.70% of total dataset).
- **Key Characteristics:**
  - Highest average purchase price per property (**$350,987.23** per unit).
  - Highest corporate client representation (**7.3% Company clients**).
  - Highest price-per-sqft tolerance (**$303.36 / sqft**).
- **Marketing Implications:** Offer high-end luxury residential and commercial office space; tailor corporate B2B real estate acquisition channels.

---

### D. Cluster 3: Leveraged Investment Buyers
- **Size:** 222 buyers (11.10% of total dataset).
- **Key Characteristics:**
  - Highest mortgage loan utilization (**42.79%** loan application rate).
  - Highest investment purpose ratio (**32.88%** investment intent).
  - Active across international markets (USA, UK, Canada, France, Mexico).
- **Marketing Implications:** Partner with financial institutions to offer attractive mortgage packages and ROI-focused investment property listings.

---

## 4. Visualizations Created

1. **Buyer Distribution (`reports/cluster_distribution.png`):** Bar plot illustrating segment buyer counts and dataset proportions.
2. **Characteristics Comparison Dashboard (`reports/cluster_characteristics.png`):** 4-panel dashboard comparing Total Spend, Unit Price, Loan Utilization, and Investment Intent across clusters.

---
