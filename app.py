import streamlit as st
import plotly.express as px

from src.data.loader import DataLoader

from src.features.engagement import EngagementFeature
from src.features.premium_customer import PremiumCustomer
from src.features.relationship_score import RelationshipScore
from src.features.sticky_customer import StickyCustomer

from src.kpi.kpi_metrics import KPIMetrics


st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon="🏦",
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

# SIDEBAR

st.sidebar.title("Filters")

country = st.sidebar.multiselect(
    "Country",
    df["Geography"].unique(),
    default=df["Geography"].unique()
)

gender = st.sidebar.multiselect(
    "Gender",
    df["Gender"].unique(),
    default=df["Gender"].unique()
)

product = st.sidebar.multiselect(
    "Products",
    sorted(df["NumOfProducts"].unique()),
    default=sorted(df["NumOfProducts"].unique())
)

df = df[
    (df["Geography"].isin(country))
    &
    (df["Gender"].isin(gender))
    &
    (df["NumOfProducts"].isin(product))
]

# TITLE

st.title("🏦 Customer Engagement & Retention Analytics")

st.markdown("---")

# KPI

c1,c2,c3,c4,c5 = st.columns(5)

c1.metric(
    "Customers",
    len(df)
)

c2.metric(
    "Churn Rate",
    f"{KPIMetrics.overall_churn(df):.2f}%"
)

c3.metric(
    "Average RSI",
    f"{KPIMetrics.average_rsi(df):.2f}"
)

c4.metric(
    "Sticky",
    f"{KPIMetrics.sticky_rate(df):.2f}%"
)

premium = round(
    df["HighValueCustomer"].mean()*100,
    2
)

c5.metric(
    "Premium",
    f"{premium}%"
)

st.markdown("---")

# CHARTS

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

    profile = df[
        "EngagementProfile"
    ].value_counts().reset_index()

    profile.columns = [
        "Profile",
        "Count"
    ]

    fig = px.bar(
        profile,
        x="Profile",
        y="Count",
        title="Engagement Profiles"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    fig = px.histogram(
        df,
        x="RSI",
        nbins=25,
        title="Relationship Strength Index"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

product_df = df.groupby(
    "NumOfProducts"
)["Exited"].mean().reset_index()

product_df["Exited"] *= 100

fig = px.line(
    product_df,
    x="NumOfProducts",
    y="Exited",
    markers=True,
    title="Product Utilization vs Churn"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")

left,right = st.columns(2)

with left:

    premium_df = df[
        "HighValueCustomer"
    ].value_counts().reset_index()

    premium_df.columns = [
        "Type",
        "Count"
    ]

    premium_df["Type"] = premium_df[
        "Type"
    ].map({
        1:"Premium",
        0:"Regular"
    })

    fig = px.pie(
        premium_df,
        names="Type",
        values="Count",
        title="Premium Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    sticky = df[
        "StickyCustomer"
    ].value_counts().reset_index()

    sticky.columns = [
        "Sticky",
        "Count"
    ]

    sticky["Sticky"] = sticky[
        "Sticky"
    ].map({
        1:"Sticky",
        0:"Normal"
    })

    fig = px.pie(
        sticky,
        names="Sticky",
        values="Count",
        title="Sticky Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

st.subheader("Business Insights")

st.success(
    f"""
    • Customer Churn Rate : {KPIMetrics.overall_churn(df):.2f}%

    • Average Relationship Strength : {KPIMetrics.average_rsi(df):.2f}

    • Sticky Customer Rate : {KPIMetrics.sticky_rate(df):.2f}%

    • Premium Customer Rate : {premium}%
    """
)

st.markdown("---")

with st.expander("View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )