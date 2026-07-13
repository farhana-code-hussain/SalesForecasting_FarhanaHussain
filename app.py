import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load Datasets
# ----------------------------

@st.cache_data
def load_data():

    sales_df = pd.read_csv("train.csv")

    games_df = pd.read_csv("vgsales.csv")

    sales_df["Order Date"] = pd.to_datetime(
        sales_df["Order Date"],
        dayfirst=True,
        format="mixed"
)

    sales_df["Ship Date"] = pd.to_datetime(
        sales_df["Ship Date"],
        dayfirst=True,
        format="mixed"
)
# Create derived columns used by the dashboard

    sales_df["Year"] = sales_df["Order Date"].dt.year

    sales_df["Month"] = sales_df["Order Date"].dt.month_name()

    sales_df["Quarter"] = sales_df["Order Date"].dt.quarter

    sales_df["Week"] = sales_df["Order Date"].dt.isocalendar().week.astype(int)

    sales_df["DayOfWeek"] = sales_df["Order Date"].dt.day_name()

    sales_df["Shipping Days"] = (
        sales_df["Ship Date"] - sales_df["Order Date"]
    ).dt.days

    return sales_df, games_df


sales_df, games_df = load_data()

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("📊 Dashboard Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Home",
        "📂 Dataset Overview",
        "📈 Sales Analysis",
        "🔮 Forecasting",
        "🚨 Anomaly Detection",
        "🎯 Product Segmentation",
        "🖼 Charts Gallery",
        "ℹ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("Project Status")

st.sidebar.write("✔ Data Analysis")

st.sidebar.write("✔ Forecasting")

st.sidebar.write("✔ Anomaly Detection")

st.sidebar.write("✔ Clustering")

st.sidebar.write("✔ Dashboard")

st.sidebar.markdown("---")

st.sidebar.info(
    "Internship Project\n\nPrepared by Farhana Hussain"
)

# ----------------------------
# HOME PAGE
# ----------------------------

if page == "🏠 Home":

    st.title("📈 Sales Forecasting Dashboard")

    st.subheader("Internship Project")

    st.write("Prepared by **Farhana Hussain**")

    st.markdown("---")

    st.header("Project Overview")

    st.write(
        """
This project performs complete Sales Forecasting and Business Analytics using Machine Learning.

The dashboard includes:

- 📊 Sales Data Analysis

- 📈 Time Series Forecasting

- 🔮 SARIMA Forecast

- 🤖 Prophet Forecast

- ⚡ XGBoost Forecast

- 🚨 Isolation Forest Anomaly Detection

- 📉 Z-Score Anomaly Detection

- 🎯 Product Demand Segmentation

- 📦 Inventory Recommendations
"""
    )

    st.markdown("---")

    st.header("Project Workflow")

    st.write(
        """
1. Data Collection

2. Data Cleaning

3. Exploratory Data Analysis

4. Time Series Analysis

5. Forecasting Models

6. Model Comparison

7. Anomaly Detection

8. Product Clustering

9. Dashboard Development
"""
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Sales Dataset Rows",
            sales_df.shape[0]
        )

        st.metric(
            "Sales Dataset Columns",
            sales_df.shape[1]
        )

    with col2:

        st.metric(
            "Video Game Dataset Rows",
            games_df.shape[0]
        )

        st.metric(
            "Video Game Dataset Columns",
            games_df.shape[1]
        )

    st.success("Dashboard Loaded Successfully ✅")

    # ----------------------------
# DATASET OVERVIEW
# ----------------------------

elif page == "📂 Dataset Overview":

    st.title("📂 Dataset Overview")
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Sales Rows",
        sales_df.shape[0]
    )

    col2.metric(
        "Game Records",
        games_df.shape[0]
    )

    col3.metric(
        "Sales Columns",
        sales_df.shape[1]
    )

    tab1, tab2 = st.tabs(
        ["Sales Dataset", "Video Games Dataset"]
    )

    # ---------------------------------

    with tab1:

        st.subheader("Sales Dataset")

        st.write("Shape :", sales_df.shape)

        st.write("Columns")

        st.dataframe(
            pd.DataFrame(
                sales_df.columns,
                columns=["Column Name"]
            )
        )

        st.subheader("First 10 Records")

        st.dataframe(
            sales_df.head(10),
            use_container_width=True
        )

        st.subheader("Summary Statistics")

        st.dataframe(
            sales_df.describe(),
            use_container_width=True
        )

    # ---------------------------------

    with tab2:

        st.subheader("Video Games Dataset")

        st.write("Shape :", games_df.shape)

        st.write("Columns")

        st.dataframe(
            pd.DataFrame(
                games_df.columns,
                columns=["Column Name"]
            )
        )

        st.subheader("First 10 Records")

        st.dataframe(
            games_df.head(10),
            use_container_width=True
        )

        st.subheader("Summary Statistics")

        st.dataframe(
            games_df.describe(),
            use_container_width=True
        )

# ----------------------------
# SALES ANALYSIS
# ----------------------------

elif page == "📈 Sales Analysis":

    st.title("📈 Sales Analysis")

    st.write(
        "Interactive analysis of sales and video game datasets."
    )

    st.markdown("---")
    st.subheader("📅 Monthly Sales Trend")

    # Create Month column from Order Date
    sales_df["Month"] = sales_df["Order Date"].dt.month_name()

    month_order = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]

    monthly_sales = (
        sales_df.groupby("Month")["Sales"]
        .sum()
        .reindex(month_order)
    )

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        monthly_sales.index,
        monthly_sales.values,
        marker="o",
        linewidth=2
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.set_title("Monthly Sales")

    st.pyplot(fig)
    st.markdown("---")

    st.subheader("🌍 Sales by Region")

    region_sales = (
        sales_df.groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9,5))

    region_sales.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Sales")

    st.pyplot(fig)
    st.markdown("---")

    st.subheader("📦 Revenue by Category")

    category_sales = (
        sales_df.groupby("Category")["Sales"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(8,5))

    category_sales.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Revenue")

    st.pyplot(fig)
    st.markdown("---")

    st.subheader("🚚 Average Shipping Days")

    shipping = (
        sales_df.groupby("Region")["Shipping Days"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8,5))

    shipping.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Average Days")

    st.pyplot(fig)
    st.markdown("---")

    st.subheader("🎮 Top 10 Video Game Genres")

    genre_sales = (
        games_df.groupby("Genre")["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,5))

    genre_sales.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Global Sales")

    st.pyplot(fig)
    st.markdown("---")

    st.subheader("🕹 Top 10 Platforms")

    platform_sales = (
        games_df.groupby("Platform")["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,5))

    platform_sales.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Global Sales")

    st.pyplot(fig)
# ----------------------------
# FORECASTING
# ----------------------------

elif page == "🔮 Forecasting":

    st.title("🔮 Sales Forecasting")

    st.write(
        """
This section presents the forecasting models developed during the project.

The following forecasting techniques were implemented:

• Prophet

• SARIMA

• XGBoost

• Forecast Comparison
"""
    )

    st.markdown("---")

    charts_path = "charts"

    forecast_images = [
        ("Prophet Forecast", "prophet_forecast.png"),
        ("SARIMA Forecast", "sarima_forecast.png"),
        ("XGBoost Forecast", "xgboost_forecast.png"),
        ("Forecast Model Comparison", "prophet_components.png")
    ]

    for title, filename in forecast_images:

        filepath = os.path.join(charts_path, filename)

        st.subheader(title)

        if os.path.exists(filepath):

            st.image(
                filepath,
                use_container_width=True
            )

        else:

            st.warning(f"{filename} not found.")

        st.markdown("---")
# ----------------------------
# ANOMALY DETECTION
# ----------------------------

elif page == "🚨 Anomaly Detection":

    st.title("🚨 Anomaly Detection")

    st.write("""
This section shows anomalies detected using Machine Learning techniques.

Methods Used:
- Isolation Forest
- Z-Score Analysis
""")

    st.markdown("---")

    charts_path = "charts"

    anomaly_images = [
        ("Isolation Forest Anomalies", "isolation_forest_anomalies.png"),
        ("Z-Score Anomalies", "zscore_anomalies.png")
    ]

    for title, filename in anomaly_images:

        st.subheader(title)

        filepath = os.path.join(charts_path, filename)

        if os.path.exists(filepath):
            st.image(filepath, use_container_width=True)
        else:
            st.warning(f"{filename} not found.")

        st.markdown("---")
# ----------------------------
# PRODUCT SEGMENTATION
# ----------------------------

elif page == "🎯 Product Segmentation":

    st.title("🎯 Product Segmentation")

    st.write("""
Product segmentation was performed using K-Means Clustering.

Visualizations below show:

- Elbow Method
- Product Clusters
""")

    st.markdown("---")

    charts_path = "charts"

    cluster_images = [
        ("Elbow Method", "elbow_method.png"),
        ("Product Clusters", "product_clusters.png")
    ]

    for title, filename in cluster_images:

        st.subheader(title)

        filepath = os.path.join(charts_path, filename)

        if os.path.exists(filepath):
            st.image(filepath, use_container_width=True)
        else:
            st.warning(f"{filename} not found.")

        st.markdown("---")
# ----------------------------
# CHARTS GALLERY
# ----------------------------

elif page == "🖼 Charts Gallery":

    st.title("🖼 Charts Gallery")

    st.write(
        "All charts generated during the project are displayed below."
    )

    charts_path = "charts"

    chart_files = sorted(
        [
            f for f in os.listdir(charts_path)
            if f.endswith(".png")
        ]
    )

    if len(chart_files) == 0:

        st.warning("No charts found.")

    else:

        cols = st.columns(2)

        for i, chart in enumerate(chart_files):

            filepath = os.path.join(charts_path, chart)

            with cols[i % 2]:

                st.image(
                    filepath,
                    caption=chart.replace(".png", "").replace("_", " ").title(),
                    use_container_width=True
                )

# ----------------------------
# ABOUT PROJECT
# ----------------------------

elif page == "ℹ About Project":

    st.title("ℹ About Project")

    st.markdown("""
## 📈 Sales Forecasting Dashboard

### Prepared By

**Farhana Hussain**

---

### Project Description

This project demonstrates an end-to-end Machine Learning workflow for
Sales Forecasting and Business Analytics.

The project includes:

- Exploratory Data Analysis (EDA)
- Time Series Forecasting
- SARIMA Forecasting
- Prophet Forecasting
- XGBoost Forecasting
- Isolation Forest Anomaly Detection
- Z-Score Analysis
- Product Segmentation using K-Means Clustering
- Interactive Streamlit Dashboard

---

### Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Prophet
- XGBoost
- Streamlit

---

### Datasets

- Retail Sales Dataset
- Video Games Sales Dataset

---

### Internship Submission

This dashboard was developed as part of the Data Science Internship project.

© 2026 Farhana Hussain
""")
    
# ----------------------------
# FOOTER
# ----------------------------

st.markdown("---")

st.caption(
    "Developed by Farhana Hussain | Sales Forecasting Dashboard | 2026"
)