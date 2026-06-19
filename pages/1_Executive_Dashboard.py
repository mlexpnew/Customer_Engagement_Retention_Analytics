import streamlit as st
import plotly.express as px

from src.data.loader import DataLoader
from src.features.engagement import EngagementFeature
from src.features.premium_customer import PremiumCustomer
from src.features.relationship_score import RelationshipScore
from src.features.sticky_customer import StickyCustomer
from src.kpi.kpi_metrics import KPIMetrics


st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():

    loader = DataLoader()

    df = loader.load_data()

    df = EngagementFeature.create(df)
    df = PremiumCustomer.create(df)
    df = RelationshipScore.create(df)
    df = StickyCustomer.create(df)

    return df


df = load_data()

st.title("📊 Executive Dashboard")

st.markdown("---")

# KPI Cards

col1,col2,col3,col4,col5 = st.columns(5)

customers = len(df)

churn = KPIMetrics.overall_churn(df)

avg_rsi = KPIMetrics.average_rsi(df)

sticky = KPIMetrics.sticky_rate(df)

premium = round(
    df["HighValueCustomer"].mean()*100,
    2
)

col1.metric(
    "Customers",
    customers
)

col2.metric(
    "Churn Rate",
    f"{churn:.2f}%"
)

col3.metric(
    "Average RSI",
    f"{avg_rsi:.2f}"
)

col4.metric(
    "Sticky",
    f"{sticky:.2f}%"
)

col5.metric(
    "Premium",
    f"{premium:.2f}%"
)

st.markdown("---")

# Charts

left,right = st.columns(2)

with left:

    fig = px.pie(
        df,
        names="Exited",
        title="Customer Churn"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    geo = df.groupby(
        "Geography"
    )["Exited"].mean().reset_index()

    geo["Exited"] *= 100

    fig = px.bar(
        geo,
        x="Geography",
        y="Exited",
        title="Geography Churn %"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

left,right = st.columns(2)

with left:

    gender = df.groupby(
        "Gender"
    )["Exited"].mean().reset_index()

    gender["Exited"] *= 100

    fig = px.bar(
        gender,
        x="Gender",
        y="Exited",
        title="Gender Churn %"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    fig = px.histogram(
        df,
        x="Age",
        nbins=20,
        title="Age Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

st.subheader("Executive Summary")

germany = round(
    df[
        df["Geography"]=="Germany"
    ]["Exited"].mean()*100,
    2
)

france = round(
    df[
        df["Geography"]=="France"
    ]["Exited"].mean()*100,
    2
)

active = round(
    df[
        df["IsActiveMember"]==1
    ]["Exited"].mean()*100,
    2
)

inactive = round(
    df[
        df["IsActiveMember"]==0
    ]["Exited"].mean()*100,
    2
)

st.success(f"""

### Key Business Insights

• Total Customers : {customers}

• Overall Churn Rate : {churn:.2f}%

• Germany Churn : {germany:.2f}%

• France Churn : {france:.2f}%

• Active Customer Churn : {active:.2f}%

• Inactive Customer Churn : {inactive:.2f}%

• Premium Customers : {premium:.2f}%

• Sticky Customers : {sticky:.2f}%

""")

st.markdown("---")

with st.expander("View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )