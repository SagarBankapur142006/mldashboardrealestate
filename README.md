# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

## 📌 Project Overview
Production-grade machine learning and market intelligence solution designed to segment real estate buyers, profile investment portfolios, and deliver actionable marketing strategies for real estate intelligence.

---

## 🏗 Project Architecture & Structure

```
d:\mlintern\
├── clients.csv                     # Raw Clients Dataset (UNTOUCHED)
├── properties.csv                  # Raw Properties Dataset (UNTOUCHED)
├── requirements.txt                # Python project dependencies
├── README.md                       # Complete setup and execution documentation
├── app.py                          # Streamlit Interactive Web Application Entry Point
├── step1_data_cleaning.py          # Step 1: Data Cleaning & Normalization Script
├── step2_feature_encoding.py       # Step 2: Categorical Feature Encoding Script
├── step3_feature_scaling.py        # Step 3: Numerical Feature Scaling Script
├── data/                           # Processed & Clustered Datasets
│   ├── cleaned_clients.csv
│   ├── cleaned_properties.csv
│   ├── encoded_clients.csv
│   ├── scaled_clients.csv
│   ├── clustered_clients.csv
│   └── cluster_profiles.csv
├── src/                            # Modular Python Source Code Package
│   ├── __init__.py
│   ├── utils.py                    # Constants, path configurations, helper functions
│   ├── data_loader.py              # Dataset loading & schema validation
│   ├── preprocessor.py             # Preprocessing & feature engineering pipeline
│   ├── eda.py                      # Exploratory Data Analysis engine
│   ├── clustering.py               # Step 4: Clustering Model Selection (K-Means & Hierarchical)
│   ├── cluster_evaluation.py       # Step 5: Optimal Cluster Evaluation (Elbow & Silhouette)
│   └── cluster_interpretation.py   # Step 6: Cluster Interpretation & Buyer Profiling
└── reports/                        # Automated Markdown Reports & Visual Charts
    ├── data_quality_report.md
    ├── eda_report.md
    ├── cluster_evaluation.csv
    ├── cluster_interpretation.md
    ├── elbow_method.png
    ├── silhouette_scores.png
    ├── cluster_distribution.png
    └── cluster_characteristics.png
```

---

## 🚀 Setup & Execution Instructions

### 1. Environment Setup
Ensure Python 3.9+ is installed. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

---

### 2. Execution Pipeline (Step-by-Step)

Each step can be executed independently using simple Python commands:

- **Step 1: Data Cleaning & Normalization**
  ```bash
  python step1_data_cleaning.py
  ```
- **Step 2: Feature Encoding**
  ```bash
  python step2_feature_encoding.py
  ```
- **Step 3: Feature Scaling**
  ```bash
  python step3_feature_scaling.py
  ```
- **Step 4: Clustering Model Selection**
  ```bash
  python src/clustering.py
  ```
- **Step 5: Optimal Cluster Selection**
  ```bash
  python src/cluster_evaluation.py
  ```
- **Step 6: Cluster Interpretation & Profiling**
  ```bash
  python src/cluster_interpretation.py
  ```

---

### 3. Launching the Interactive Streamlit Dashboard (Step 7)

Launch the interactive real estate market intelligence application:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🖥 Application Sections

1. **📊 Buyer Segmentation Overview:** KPI metrics, total buyers count, cluster volume bar/pie charts, and interactive buyer data table.
2. **📈 Investor Behavior Dashboard:** Acquisition purpose distribution (Home vs Investment), loan utilization, client type breakdown, and Sold property transaction stats (Total Spend, Avg Unit Price, Avg Floor Area, Price/sqft).
3. **🌐 Geographic Buyer Analysis:** World Choropleth heatmap, country buyer counts, and top 15 region breakdown.
4. **💡 Segment Insights & Marketing Panel:** Detailed empirical characteristics, buyer demographics, and actionable marketing strategies for all 4 clusters.
5. **🔍 Interactive Sidebar Filters:** Country, Region, Acquisition Purpose, and Client Type multiselect controls.
