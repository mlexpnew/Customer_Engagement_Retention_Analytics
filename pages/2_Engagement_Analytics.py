import streamlit as st
import plotly.express as px

from src.data.loader import DataLoader
from src.features.engagement import EngagementFeature
from src.features.premium_customer import PremiumCustomer
from src.features.relationship_score import RelationshipScore
from src.features.sticky_customer import StickyCustomer

st.set_page_config(
    page_title="Engagement Analytics",
    page_icon="👥",
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

st.title("👥 Engagement Analytics")

st.markdown("---")

# KPI

c1,c2,c3,c4 = st.columns(4)

active = round(
    (df["IsActiveMember"] == 1).mean()*100,
    2
)

inactive = round(
    (df["IsActiveMember"] == 0).mean()*100,
    2
)

avg_rsi = round(
    df["RSI"].mean(),
    2
)

sticky = round(
    df["StickyCustomer"].mean()*100,
    2
)

c1.metric(
    "Active",
    f"{active}%"
)

c2.metric(
    "Inactive",
    f"{inactive}%"
)

c3.metric(
    "Average RSI",
    avg_rsi
)

c4.metric(
    "Sticky",
    f"{sticky}%"
)

st.markdown("---")

# Engagement Profiles

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

    profile_churn = df.groupby(
        "EngagementProfile"
    )["Exited"].mean().reset_index()

    profile_churn["Exited"] *= 100

    fig = px.bar(
        profile_churn,
        x="EngagementProfile",
        y="Exited",
        title="Profile Churn %"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

# RSI

left,right = st.columns(2)

with left:

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

with right:

    sticky_df = df[
        "StickyCustomer"
    ].value_counts().reset_index()

    sticky_df.columns = [
        "Sticky",
        "Count"
    ]

    sticky_df["Sticky"] = sticky_df[
        "Sticky"
    ].map({
        1:"Sticky",
        0:"Normal"
    })

    fig = px.pie(
        sticky_df,
        names="Sticky",
        values="Count",
        title="Sticky Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

st.subheader("Engagement Insights")

profile_summary = df[
    "EngagementProfile"
].value_counts()

st.info(f"""

### Key Findings

• Active Customers : {active}%

• Inactive Customers : {inactive}%

• Average Relationship Strength : {avg_rsi}

• Sticky Customers : {sticky}%

• Largest Segment : {profile_summary.index[0]}

• Smallest Segment : {profile_summary.index[-1]}

""")

st.markdown("---")

with st.expander("View Engagement Data"):

    st.dataframe(
        df[
            [
                "CustomerId",
                "EngagementProfile",
                "RSI",
                "StickyCustomer",
                "Exited"
            ]
        ],
        use_container_width=True
    )