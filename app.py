import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# App Settings
# -------------------------------
st.set_page_config(page_title="Malaria Predictor Dashboard", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-color: #eef5f9;
    }
    .stRadio > div {
        background-color: #dceefa;
        padding: 10px;
        border-radius: 10px;
    }
    .block-container {
        padding-top: 1rem;
        background-color: #fdf9fc;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
    }
    .stButton > button:hover {
        background-color: #105e8b;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🦟 Malaria Case Predictor")

# -------------------------------
# Load Data and Model
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("malaria_africa_cleaned.csv")
    df["Year"] = df["Year"].astype(int)
    return df

def load_model():
    return joblib.load("model_compatible.pkl")

df = load_data()
model = load_model()

# Add ISO Alpha-3 codes to the dataframe for mapping
import pycountry

def get_iso_alpha(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except LookupError:
        return None

df["iso_alpha"] = df["Country"].apply(get_iso_alpha)

# -------------------------------
# Session State for Tabs
# -------------------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Dashboard"

tab_labels = ["📊 Dashboard", "🎯 Try Prediction", "📈 Trends & Insights"]
selected_tab = st.radio("Navigation", tab_labels, index=tab_labels.index(st.session_state.active_tab), horizontal=True)
st.session_state.active_tab = selected_tab

# Setup target and features
target = "No. of cases_median"
numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col != target]

# -------------------------------
# Dashboard Tab
# -------------------------------
if selected_tab == "📊 Dashboard":
    st.subheader("🔍 Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        total_cases = int(df["No. of cases_median"].sum())
        st.metric("Total Malaria Cases (Median)", f"{total_cases:,}")

    with col2:
        latest_year = df["Year"].max()
        st.metric("Latest Year in Data", latest_year)

    with col3:
        total_deaths = int(df["No. of deaths_median"].sum())
        st.metric("Total Deaths (Median)", f"{total_deaths:,}")

    st.markdown("---")
    st.subheader("🌍 Top 10 Countries by Cases in Latest Year")
    latest_df = df[df["Year"] == latest_year].copy()
    latest_df["No. of cases_median"] = pd.to_numeric(latest_df["No. of cases_median"], errors="coerce")
    top10 = latest_df.groupby("Country")["No. of cases_median"].sum().sort_values(ascending=False).head(10)
    fig = px.bar(top10, x=top10.values, y=top10.index, orientation='h', labels={'x': 'Cases'}, title="Top 10 Countries")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗺️ Malaria Map (Latest Year)")
    if "iso_alpha" not in df.columns:
        import pycountry
        country_map = {country.name: country.alpha_3 for country in pycountry.countries}
        df["iso_alpha"] = df["Country"].map(country_map)
    map_df = latest_df.dropna(subset=["iso_alpha"])
    map_fig = px.choropleth(map_df,
                            locations="iso_alpha",
                            color="No. of cases_median",
                            hover_name="Country",
                            color_continuous_scale="Reds",
                            title="Malaria Cases by Country (Choropleth Map)")
    st.plotly_chart(map_fig, use_container_width=True)

# -------------------------------
# Try Prediction Tab
# -------------------------------
elif selected_tab == "🎯 Try Prediction":
    st.subheader("🎯 Predict Malaria Cases")

    input_data = {}
    with st.form("predict_form"):
        for col in numeric_cols:
            if col == "Year":
                min_val = int(df[col].min())
                max_val = int(df[col].max())
                mean_val = int(df[col].mean())
                input_data[col] = st.slider(col, min_val, max_val, mean_val, step=1)
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                input_data[col] = st.slider(col, min_val, max_val, mean_val)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.session_state.predicted_result = prediction

    if "predicted_result" in st.session_state:
        st.success(f"🧪 Predicted malaria cases: {st.session_state.predicted_result:,.0f}")

# -------------------------------
# Trends & Insights Tab
# -------------------------------
elif selected_tab == "📈 Trends & Insights":
    st.subheader("📈 Malaria Trends Over Time")

    country = st.selectbox("Choose a country:", sorted(df["Country"].unique()))
    country_df = df[df["Country"] == country].copy()

    if country_df.empty:
        st.warning("No data found for this country.")
    else:
        line_fig = px.line(country_df, x="Year", y="No. of cases_median", markers=True,
                           title=f"Yearly Malaria Cases (Median) for {country}")
        st.plotly_chart(line_fig, use_container_width=True)

        # Pie chart of cases vs deaths
        total_cases = country_df["No. of cases_median"].sum()
        total_deaths = country_df["No. of deaths_median"].sum()

        pie_data = pd.DataFrame({
            "Type": ["Cases", "Deaths"],
            "Count": [total_cases, total_deaths]
        })
        pie_fig = px.pie(pie_data, values="Count", names="Type", title=f"Cases vs Deaths in {country}",
                         color_discrete_sequence=["#2ca02c", "#d62728"])
        st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Model Performance")
    X = df[numeric_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    st.write("**Evaluation Metrics:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.0f}")
    col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.0f}")
    col3.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
    col4.metric("Data Points", len(df))
