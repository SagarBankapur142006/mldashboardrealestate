import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Real Estate Buyer Segmentation & Market Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern real estate dashboard theme
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. LOAD DATASETS SAFELY WITH RELATIVE PATHS
# ----------------------------------------------------
@st.cache_data
def load_datasets():
    data_dir = "data"
    clustered_path = os.path.join(data_dir, "clustered_clients.csv")
    properties_path = os.path.join(data_dir, "cleaned_properties.csv")
    profiles_path = os.path.join(data_dir, "cluster_profiles.csv")

    if not os.path.exists(clustered_path) or not os.path.exists(properties_path):
        st.error("Error: Required dataset files missing from data/ directory. Please run Step 1 - Step 6 scripts first.")
        st.stop()

    clients = pd.read_csv(clustered_path)
    properties = pd.read_csv(properties_path)
    profiles = pd.read_csv(profiles_path) if os.path.exists(profiles_path) else None

    # Process Sold properties (exclude Available listings from purchase transactions)
    sold_props = properties[properties["listing_status"] == "Sold"].copy()
    sold_props["sale_price_clean"] = (
        sold_props["sale_price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    sold_props["price_per_sqft"] = sold_props["sale_price_clean"] / sold_props["floor_area_sqft"]

    # Aggregate Sold property transactions per buyer
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

    # Merge client profiles with property portfolio metrics
    merged = pd.merge(clients, buyer_props, left_on="client_id", right_on="client_ref", how="left")

    # Mapping segment names
    segment_names = {
        0: "Segment 0: Low-Leverage Residential Buyers",
        1: "Segment 1: High-Volume Core Buyers (Global Investors)",
        2: "Segment 2: High-Valuation Premium & Corporate Buyers",
        3: "Segment 3: Leveraged Investment Buyers"
    }
    merged["segment_name"] = merged["kmeans_cluster"].map(segment_names)

    return clients, properties, sold_props, merged, profiles, segment_names

clients_df, properties_df, sold_props_df, merged_df, profiles_df, segment_names = load_datasets()

# ----------------------------------------------------
# 2. SIDEBAR NAVIGATION & USER CONTROLS (FILTERS)
# ----------------------------------------------------
st.sidebar.title("🏢 Real Estate Intelligence")
st.sidebar.markdown("**Internship Project Dashboard**")

section = st.sidebar.radio(
    "Navigate Section:",
    [
        "📊 Buyer Segmentation Overview",
        "📈 Investor Behavior Dashboard",
        "🌐 Geographic Buyer Analysis",
        "💡 Segment Insights & Marketing Panel"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Interactive Sidebar Filters")

# Filter Controls
selected_countries = st.sidebar.multiselect(
    "Filter by Country:",
    options=sorted(merged_df["country"].dropna().unique().tolist()),
    default=sorted(merged_df["country"].dropna().unique().tolist())
)

available_regions = sorted(merged_df[merged_df["country"].isin(selected_countries)]["region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Filter by Region:",
    options=available_regions,
    default=available_regions
)

selected_purposes = st.sidebar.multiselect(
    "Filter by Acquisition Purpose:",
    options=sorted(merged_df["acquisition_purpose"].unique().tolist()),
    default=sorted(merged_df["acquisition_purpose"].unique().tolist())
)

selected_types = st.sidebar.multiselect(
    "Filter by Client Type:",
    options=sorted(merged_df["client_type"].unique().tolist()),
    default=sorted(merged_df["client_type"].unique().tolist())
)

# Apply reactive filters
filtered_df = merged_df[
    (merged_df["country"].isin(selected_countries)) &
    (merged_df["region"].isin(selected_regions)) &
    (merged_df["acquisition_purpose"].isin(selected_purposes)) &
    (merged_df["client_type"].isin(selected_types))
].copy()

# Header display
st.markdown('<div class="main-title">Machine Learning Based Buyer Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Investment Profiling & Market Intelligence Dashboard</div>', unsafe_allow_html=True)

# Filter status banner
if len(filtered_df) < len(merged_df):
    st.info(f"Filtering Applied: Showing **{len(filtered_df):,}** of **{len(merged_df):,}** total buyers.")

# ----------------------------------------------------
# SECTION 1: BUYER SEGMENTATION OVERVIEW
# ----------------------------------------------------
if section == "📊 Buyer Segmentation Overview":
    st.header("📊 Buyer Segmentation Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Filtered Buyers", f"{len(filtered_df):,}")
    
    total_spend = filtered_df["total_spend"].sum()
    col2.metric("Cumulative Portfolio Spend", f"${total_spend:,.0f}")
    
    avg_spend = filtered_df["total_spend"].mean() if len(filtered_df) > 0 else 0
    col3.metric("Avg Spend per Buyer", f"${avg_spend:,.0f}")
    
    avg_props = filtered_df["properties_bought"].mean() if len(filtered_df) > 0 else 0
    col4.metric("Avg Properties / Buyer", f"{avg_props:.2f}")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Cluster Buyer Share Distribution")
        cluster_counts = filtered_df["segment_name"].value_counts().reset_index()
        cluster_counts.columns = ["segment_name", "count"]
        
        fig_pie = px.pie(
            cluster_counts,
            names="segment_name",
            values="count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Proportion of Buyers per Cluster"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.subheader("Buyer Counts Across Clusters")
        fig_bar = px.bar(
            cluster_counts,
            x="segment_name",
            y="count",
            color="segment_name",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text="count",
            title="Absolute Buyer Volume per Cluster"
        )
        fig_bar.update_layout(showlegend=False, xaxis_title="Segment", yaxis_title="Number of Buyers")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Filtered Buyer Dataset Table")
    st.dataframe(
        filtered_df[[
            "client_id", "segment_name", "client_type", "country", "region", 
            "acquisition_purpose", "loan_applied", "age", "satisfaction_score",
            "total_spend", "avg_property_price", "properties_bought"
        ]].head(100),
        use_container_width=True
    )

# ----------------------------------------------------
# SECTION 2: INVESTOR BEHAVIOR DASHBOARD
# ----------------------------------------------------
elif section == "📈 Investor Behavior Dashboard":
    st.header("📈 Investor Behavior & Property Purchase Analysis")
    st.caption("Note: All property transaction statistics are computed strictly using **Sold** listings. Unsold Available listings are excluded from purchase calculations.")

    col1, col2, col3, col4 = st.columns(4)
    avg_unit_price = filtered_df["avg_property_price"].mean() if len(filtered_df) > 0 else 0
    col1.metric("Avg Unit Purchase Price", f"${avg_unit_price:,.0f}")
    
    avg_sqft = filtered_df["avg_sqft"].mean() if len(filtered_df) > 0 else 0
    col2.metric("Avg Floor Area (sqft)", f"{avg_sqft:.1f} sqft")
    
    avg_price_sqft = filtered_df["avg_price_per_sqft"].mean() if len(filtered_df) > 0 else 0
    col3.metric("Avg Price / Sqft", f"${avg_price_sqft:.2f}")

    loan_pct = (filtered_df["loan_applied"] == "Yes").mean() * 100 if len(filtered_df) > 0 else 0
    col4.metric("Loan Applied Ratio", f"{loan_pct:.1f}%")

    st.markdown("---")

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.subheader("Acquisition Purpose (Home vs Investment)")
        purpose_df = filtered_df.groupby(["segment_name", "acquisition_purpose"]).size().reset_index(name="count")
        fig_purp = px.bar(
            purpose_df,
            x="segment_name",
            y="count",
            color="acquisition_purpose",
            barmode="group",
            title="Acquisition Purpose Breakdown by Cluster",
            color_discrete_sequence=["#2B5C8F", "#D9534F"]
        )
        st.plotly_chart(fig_purp, use_container_width=True)

    with col_b2:
        st.subheader("Loan Utilization Across Clusters")
        loan_df = filtered_df.groupby(["segment_name", "loan_applied"]).size().reset_index(name="count")
        fig_loan = px.bar(
            loan_df,
            x="segment_name",
            y="count",
            color="loan_applied",
            barmode="group",
            title="Loan Application Status by Cluster",
            color_discrete_sequence=["#5CB85C", "#F0AD4E"]
        )
        st.plotly_chart(fig_loan, use_container_width=True)

    col_b3, col_b4 = st.columns(2)

    with col_b3:
        st.subheader("Average Portfolio Spend by Segment ($)")
        spend_df = filtered_df.groupby("segment_name")["total_spend"].mean().reset_index()
        fig_spend = px.bar(
            spend_df,
            x="segment_name",
            y="total_spend",
            color="segment_name",
            title="Mean Cumulative Spend per Buyer ($)",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_spend.update_layout(showlegend=False, yaxis_tickprefix="$")
        st.plotly_chart(fig_spend, use_container_width=True)

    with col_b4:
        st.subheader("Unit Category Breakdown (Apartment vs Office)")
        unit_df = filtered_df.groupby("segment_name")[["apartment_count", "office_count"]].sum().reset_index()
        unit_melted = pd.melt(unit_df, id_vars=["segment_name"], value_vars=["apartment_count", "office_count"], var_name="unit_type", value_name="count")
        fig_unit = px.bar(
            unit_melted,
            x="segment_name",
            y="count",
            color="unit_type",
            barmode="stack",
            title="Total Units Acquired by Category",
            color_discrete_sequence=["#1F77B4", "#FF7F0E"]
        )
        st.plotly_chart(fig_unit, use_container_width=True)

# ----------------------------------------------------
# SECTION 3: GEOGRAPHIC BUYER ANALYSIS
# ----------------------------------------------------
elif section == "🌐 Geographic Buyer Analysis":
    st.header("🌐 Geographic Buyer Distribution")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Country-Level Buyer Concentration")
        country_counts = filtered_df["country"].value_counts().reset_index()
        country_counts.columns = ["country", "buyer_count"]
        
        fig_country = px.bar(
            country_counts,
            x="country",
            y="buyer_count",
            color="country",
            title="Buyer Count by Country",
            text="buyer_count"
        )
        fig_country.update_layout(showlegend=False)
        st.plotly_chart(fig_country, use_container_width=True)

    with col_g2:
        st.subheader("Global Choropleth Heatmap")
        fig_map = px.choropleth(
            country_counts,
            locations="country",
            locationmode="country names",
            color="buyer_count",
            hover_name="country",
            color_continuous_scale="Viridis",
            title="Global Geographic Distribution Map"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Top 15 Regions by Buyer Volume")
    region_counts = filtered_df["region"].value_counts().head(15).reset_index()
    region_counts.columns = ["region", "buyer_count"]
    fig_region = px.bar(
        region_counts,
        x="buyer_count",
        y="region",
        orientation="h",
        color="buyer_count",
        title="Top 15 Regions Distribution",
        color_continuous_scale="Blues"
    )
    fig_region.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_region, use_container_width=True)

# ----------------------------------------------------
# SECTION 4: SEGMENT INSIGHTS & MARKETING PANEL
# ----------------------------------------------------
elif section == "💡 Segment Insights & Marketing Panel":
    st.header("💡 Buyer Segment Insights & Marketing Strategy")

    # Render cards for each cluster
    for c in range(4):
        seg_title = segment_names[c]
        sub = filtered_df[filtered_df["kmeans_cluster"] == c]
        
        with st.expander(f"📌 {seg_title} ({len(sub)} Buyers - {len(sub)/len(filtered_df)*100:.1f}%)" if len(filtered_df)>0 else f"📌 {seg_title}", expanded=True):
            if len(sub) > 0:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Buyer Count", len(sub))
                c2.metric("Avg Buyer Age", f"{sub['age'].mean():.1f} yrs")
                c3.metric("Satisfaction Score", f"{sub['satisfaction_score'].mean():.2f}")
                c4.metric("Investment Intent", f"{(sub['acquisition_purpose']=='Investment').mean()*100:.1f}%")
                c5.metric("Loan Applied Ratio", f"{(sub['loan_applied']=='Yes').mean()*100:.1f}%")

                st.markdown("---")
                col_info1, col_info2 = st.columns(2)

                with col_info1:
                    st.markdown("**Empirical Data Characteristics:**")
                    if c == 0:
                        st.markdown("""
                        - **Low-Leverage Financing:** Lowest loan application rate (32.95%), indicating strong cash self-financing capability.
                        - **Satisfaction:** Highest customer satisfaction score (3.13 / 5.0).
                        - **Primary Intent:** 71.1% acquire properties for personal Home use.
                        - **Geographic Spread:** Concentrated in regional markets 44-56 (e.g., Texas, Utah, Virginia, Washington).
                        """)
                    elif c == 1:
                        st.markdown("""
                        - **High Volume & Scale:** Largest segment (55.75% of dataset). Highest cumulative portfolio spend ($1.27M / buyer) and highest acquisition count (3.71 properties / buyer).
                        - **Core Markets:** Major presence across US, Canada, Germany, and UK.
                        - **Asset Portfolio:** Holds 3,512 Apartments and 620 Offices.
                        """)
                    elif c == 2:
                        st.markdown("""
                        - **High Premium Valuation:** Highest average purchase price per property unit ($350,987 / unit) and price-per-sqft ($303.36 / sqft).
                        - **Corporate Presence:** Highest corporate client representation (7.3% Company clients).
                        - **Asset Focus:** Preference for high-tier residential apartments and office units.
                        """)
                    elif c == 3:
                        st.markdown("""
                        - **Leveraged Investment Focus:** Highest mortgage loan utilization rate (42.79%) and highest investment purpose ratio (32.88%).
                        - **Cross-Border Investors:** International investor cross-section across USA, UK, Canada, France, and Mexico.
                        """)

                with col_info2:
                    st.markdown("**Actionable Marketing & Business Strategies:**")
                    if c == 0:
                        st.markdown("""
                        - Target with ready-to-move-in luxury residential homes.
                        - Offer upfront cash-payment incentives or fee waivers.
                        - Emphasize post-purchase customer satisfaction and property management services.
                        """)
                    elif c == 1:
                        st.markdown("""
                        - Develop bulk-purchase multi-unit portfolio deals.
                        - Create dedicated VIP Investor communications and quarterly market reports.
                        - Offer bundled property management and tenant leasing services.
                        """)
                    elif c == 2:
                        st.markdown("""
                        - Target with premium luxury penthouses and prime commercial office real estate.
                        - Establish dedicated B2B corporate sales and institutional investment channels.
                        """)
                    elif c == 3:
                        st.markdown("""
                        - Partner with financial institutions to offer competitive mortgage packages.
                        - Highlight rental yield potential, cash-flow models, and ROI projection reports.
                        """)
            else:
                st.warning("No buyer records fit the current filter criteria for this segment.")
