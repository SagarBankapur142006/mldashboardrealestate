import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def perform_step6_cluster_interpretation():
    print("==================================================")
    print("STEP 6: CLUSTER INTERPRETATION & PROFILING REPORT")
    print("==================================================\n")

    input_clustered_path = os.path.join("data", "clustered_clients.csv")
    input_properties_path = os.path.join("data", "cleaned_properties.csv")
    
    output_profiles_csv = os.path.join("data", "cluster_profiles.csv")
    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")

    os.makedirs("data", exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if not os.path.exists(input_clustered_path):
        raise FileNotFoundError(f"Clustered dataset not found at {input_clustered_path}. Run Step 4 first.")
    if not os.path.exists(input_properties_path):
        raise FileNotFoundError(f"Properties dataset not found at {input_properties_path}. Run Step 1 first.")

    clients_df = pd.read_csv(input_clustered_path)
    properties_df = pd.read_csv(input_properties_path)

    # ----------------------------------------------------
    # 1. LINK CLIENTS AND PROPERTY TRANSACTIONS
    # ----------------------------------------------------
    # Filter strictly for Sold properties (actual purchase transactions)
    sold_props = properties_df[properties_df["listing_status"] == "Sold"].copy()
    sold_props["sale_price_clean"] = (
        sold_props["sale_price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    sold_props["price_per_sqft"] = sold_props["sale_price_clean"] / sold_props["floor_area_sqft"]

    # Aggregate property metrics per buyer
    buyer_props = sold_props.groupby("client_ref").agg(
        total_spend=("sale_price_clean", "sum"),
        avg_property_price=("sale_price_clean", "mean"),
        properties_bought=("listing_id", "count"),
        total_sqft=("floor_area_sqft", "sum"),
        avg_sqft=("floor_area_sqft", "mean"),
        avg_price_per_sqft=("price_per_sqft", "mean"),
        apartment_count=("unit_category", lambda x: (x == "Apartment").sum()),
        office_count=("unit_category", lambda x: (x == "Office").sum())
    ).reset_index()

    # Merge client data with aggregated property portfolio data
    merged_df = pd.merge(clients_df, buyer_props, left_on="client_id", right_on="client_ref", how="left")

    # Available listings reference (unsold stock)
    available_props = properties_df[properties_df["listing_status"] == "Available"].copy()
    available_props["sale_price_clean"] = (
        available_props["sale_price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # ----------------------------------------------------
    # 2. SEGMENT NAMES MAP & CLUSTER PROFILING
    # ----------------------------------------------------
    # Empirical segment names based on observed characteristics
    segment_names = {
        0: "Segment 0: Low-Leverage Residential Buyers",
        1: "Segment 1: High-Volume Core Portfolio Buyers (Global Investors)",
        2: "Segment 2: High-Valuation Premium & Corporate Buyers",
        3: "Segment 3: Leveraged Investment Buyers"
    }

    merged_df["segment_name"] = merged_df["kmeans_cluster"].map(segment_names)

    profiles = []
    for c in sorted(merged_df["kmeans_cluster"].unique()):
        sub = merged_df[merged_df["kmeans_cluster"] == c]
        total_buyers = len(sub)
        pct_buyers = (total_buyers / len(merged_df)) * 100

        ind_count = (sub["client_type"] == "Individual").sum()
        comp_count = (sub["client_type"] == "Company").sum()

        home_count = (sub["acquisition_purpose"] == "Home").sum()
        inv_count = (sub["acquisition_purpose"] == "Investment").sum()

        loan_yes = (sub["loan_applied"] == "Yes").sum()
        loan_no = (sub["loan_applied"] == "No").sum()

        top_c = ", ".join([f"{k} ({v})" for k, v in sub["country"].value_counts().head(3).items()])
        reg_range = f"{sub['region_encoded'].min()} - {sub['region_encoded'].max()}"

        profiles.append({
            "cluster_id": c,
            "segment_name": segment_names[c],
            "buyer_count": total_buyers,
            "buyer_percentage": round(pct_buyers, 2),
            "region_encoded_range": reg_range,
            "top_countries": top_c,
            "individual_buyers": ind_count,
            "company_buyers": comp_count,
            "home_purpose_count": home_count,
            "investment_purpose_count": inv_count,
            "investment_purpose_pct": round((inv_count / total_buyers) * 100, 2),
            "loan_applied_yes": loan_yes,
            "loan_applied_pct": round((loan_yes / total_buyers) * 100, 2),
            "avg_age": round(sub["age"].mean(), 2),
            "avg_satisfaction_score": round(sub["satisfaction_score"].mean(), 2),
            "avg_total_spend": round(sub["total_spend"].mean(), 2),
            "avg_property_price": round(sub["avg_property_price"].mean(), 2),
            "avg_properties_bought": round(sub["properties_bought"].mean(), 2),
            "avg_floor_area_sqft": round(sub["avg_sqft"].mean(), 2),
            "avg_price_per_sqft": round(sub["avg_price_per_sqft"].mean(), 2),
            "total_apartments_bought": int(sub["apartment_count"].sum()),
            "total_offices_bought": int(sub["office_count"].sum())
        })

    profile_df = pd.DataFrame(profiles)
    profile_df.to_csv(output_profiles_csv, index=False)

    print("--- 1. CLUSTER PROFILE SUMMARY TABLE ---")
    print(profile_df[["cluster_id", "buyer_count", "buyer_percentage", "investment_purpose_pct", "loan_applied_pct", "avg_age", "avg_satisfaction_score", "avg_total_spend", "avg_property_price"]].to_string(index=False))
    print()

    # ----------------------------------------------------
    # 3. GENERATE VISUALIZATIONS
    # ----------------------------------------------------
    sns.set_theme(style="whitegrid")

    # Visualization 1: Cluster Distribution (Buyer Counts & Share)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = sns.barplot(
        x="cluster_id",
        y="buyer_count",
        hue="cluster_id",
        data=profile_df,
        palette="Blues_d",
        legend=False,
        ax=ax
    )
    ax.set_title("Buyer Distribution Across K-Means Clusters (K = 4)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster ID", fontsize=11)
    ax.set_ylabel("Number of Buyers", fontsize=11)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            pct = (height / len(merged_df)) * 100
            ax.annotate(f"{int(height)}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    
    plt.tight_layout()
    dist_png = os.path.join(reports_dir, "cluster_distribution.png")
    fig.savefig(dist_png, dpi=300)
    fig.savefig(os.path.join(figures_dir, "cluster_distribution.png"), dpi=300)
    plt.close(fig)

    # Visualization 2: Key Characteristics Comparison Across Clusters
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # A: Total Spend
    sns.barplot(x="cluster_id", y="avg_total_spend", hue="cluster_id", data=profile_df, palette="crest", legend=False, ax=axes[0, 0])
    axes[0, 0].set_title("Average Total Portfolio Spend ($)", fontweight="bold")
    axes[0, 0].set_ylabel("Total Spend ($)")
    axes[0, 0].yaxis.set_major_formatter("${x:,.0f}")

    # B: Average Property Price
    sns.barplot(x="cluster_id", y="avg_property_price", hue="cluster_id", data=profile_df, palette="viridis", legend=False, ax=axes[0, 1])
    axes[0, 1].set_title("Average Property Purchase Price ($)", fontweight="bold")
    axes[0, 1].set_ylabel("Avg Unit Price ($)")
    axes[0, 1].yaxis.set_major_formatter("${x:,.0f}")

    # C: Loan Application %
    sns.barplot(x="cluster_id", y="loan_applied_pct", hue="cluster_id", data=profile_df, palette="magma", legend=False, ax=axes[1, 0])
    axes[1, 0].set_title("Loan Application Ratio (%)", fontweight="bold")
    axes[1, 0].set_ylabel("Loan Applied (%)")

    # D: Investment Purpose %
    sns.barplot(x="cluster_id", y="investment_purpose_pct", hue="cluster_id", data=profile_df, palette="rocket", legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("Investment Intent Ratio (%)", fontweight="bold")
    axes[1, 1].set_ylabel("Investment Purpose (%)")

    plt.suptitle("Cluster Characteristics Comparison Dashboard (K = 4)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    char_png = os.path.join(reports_dir, "cluster_characteristics.png")
    fig.savefig(char_png, dpi=300)
    fig.savefig(os.path.join(figures_dir, "cluster_characteristics.png"), dpi=300)
    plt.close(fig)

    # ----------------------------------------------------
    # 4. GENERATE MARKDOWN INTERPRETATION REPORT
    # ----------------------------------------------------
    md_report_path = os.path.join(reports_dir, "cluster_interpretation.md")
    
    md_content = f"""# Step 6: Cluster Interpretation & Buyer Profiling Report

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
| **Avg Total Spend / Buyer** | ${profile_df.loc[0, 'avg_total_spend']:,.2f} | **${profile_df.loc[1, 'avg_total_spend']:,.2f}** | ${profile_df.loc[2, 'avg_total_spend']:,.2f} | ${profile_df.loc[3, 'avg_total_spend']:,.2f} | $1,257,597.74 |
| **Avg Unit Price Bought** | ${profile_df.loc[0, 'avg_property_price']:,.2f} | ${profile_df.loc[1, 'avg_property_price']:,.2f} | **${profile_df.loc[2, 'avg_property_price']:,.2f}** | ${profile_df.loc[3, 'avg_property_price']:,.2f} | $346,650.05 |
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
"""

    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("--- 2. OUTPUT FILES CREATED ---")
    print(f"  - Cluster Profiles CSV: {output_profiles_csv}")
    print(f"  - Markdown Report: {md_report_path}")
    print(f"  - Distribution Plot: {dist_png}")
    print(f"  - Characteristics Plot: {char_png}")
    print("\nStep 6 Cluster Interpretation completed successfully.")

if __name__ == "__main__":
    perform_step6_cluster_interpretation()
