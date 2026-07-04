import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Unemployment Analysis Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
.main{background:#f5f7fa;}
[data-testid="stSidebar"]{background:#e8f0fe;}
div[data-testid="stMetric"]{background:white;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv("Unemployment.csv")
    data.columns = data.columns.str.strip()
    data.rename(columns={
        "Estimated Unemployment Rate (%)":"Unemployment_Rate",
        "Estimated Employed":"Employed",
        "Estimated Labour Participation Rate (%)":"Labour_Rate"
    }, inplace=True)

    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data.dropna(inplace=True)
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month_name()
    return data

data = load_data()

st.sidebar.title("Dashboard")
page = st.sidebar.selectbox(
    "Select Page",
    ["Overview","State Analysis","COVID-19 Analysis","Correlation","Insights"]
)

state = st.sidebar.selectbox(
    "State",
    ["All States"] + sorted(data["Region"].unique())
)

filtered = data if state=="All States" else data[data["Region"]==state]

if page=="Overview":
    st.title("📊 Unemployment Analysis Dashboard")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Records", len(filtered))
    c2.metric("Average Rate", f"{filtered['Unemployment_Rate'].mean():.2f}%")
    c3.metric("Highest", f"{filtered['Unemployment_Rate'].max():.2f}%")
    c4.metric("States", filtered["Region"].nunique())

    st.subheader("Dataset Preview")
    st.dataframe(filtered.head())

    trend = filtered.groupby("Date",as_index=False)["Unemployment_Rate"].mean()
    fig = px.line(trend,x="Date",y="Unemployment_Rate",markers=True,
                  title="Average Unemployment Trend")
    st.plotly_chart(fig,use_container_width=True)

elif page=="State Analysis":
    st.title("📈 State Analysis")

    fig = px.line(filtered,x="Date",y="Unemployment_Rate",
                  color="Region",markers=True)
    st.plotly_chart(fig,use_container_width=True)

    avg = filtered.groupby("Region",as_index=False)["Unemployment_Rate"].mean()
    fig2 = px.bar(avg,x="Region",y="Unemployment_Rate",
                  color="Unemployment_Rate",
                  title="Average Unemployment by State")
    st.plotly_chart(fig2,use_container_width=True)

elif page=="COVID-19 Analysis":
    st.title("🦠 COVID-19 Analysis")

    covid = filtered.copy()
    covid["Period"] = np.where(
        covid["Date"] >= "2020-03-01",
        "During COVID",
        "Before COVID"
    )

    summary = covid.groupby("Period",as_index=False)["Unemployment_Rate"].mean()

    fig = px.bar(summary,x="Period",y="Unemployment_Rate",
                 color="Period")
    st.plotly_chart(fig,use_container_width=True)

elif page=="Correlation":
    st.title("🌡️ Correlation Analysis")

    corr = filtered[["Unemployment_Rate","Employed","Labour_Rate"]].corr()

    fig,ax = plt.subplots(figsize=(6,4))
    sns.heatmap(corr,annot=True,cmap="viridis",ax=ax)
    st.pyplot(fig)

    scatter = px.scatter(
        filtered,
        x="Employed",
        y="Unemployment_Rate",
        color="Region",
        size="Labour_Rate",
        hover_name="Region"
    )
    st.plotly_chart(scatter,use_container_width=True)

else:
    st.title("📑 Insights")

    highest = data.groupby("Region")["Unemployment_Rate"].mean().idxmax()
    lowest = data.groupby("Region")["Unemployment_Rate"].mean().idxmin()

    st.success(f"Highest average unemployment: {highest}")
    st.success(f"Lowest average unemployment: {lowest}")

    st.markdown("""
### Recommendations
- Improve skill development.
- Encourage startups and MSMEs.
- Increase employment opportunities.
- Support youth employment programs.
- Monitor state-wise unemployment regularly.
""")

st.caption("Created by: Shiva Prasad Ghale")
