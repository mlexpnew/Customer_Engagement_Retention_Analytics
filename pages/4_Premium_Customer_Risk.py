import streamlit as st
import plotly.express as px

from src.data.loader import DataLoader
from src.features.engagement import EngagementFeature
from src.features.premium_customer import PremiumCustomer
from src.features.relationship_score import RelationshipScore
from src.features.sticky_customer import StickyCustomer

st.set_page_config(
    page_title="Premium Customer Risk",
    page_icon="👑",
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

st.title("👑 Premium Customer Risk Center")

st.markdown("---")

premium_df = df[df["HighValueCustomer"]==1]

risk_df = premium_df[
    (premium_df["IsActiveMember"]==0)
    &
    (premium_df["NumOfProducts"]<=1)
]

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Premium Customers",
    len(premium_df)
)

c2.metric(
    "At Risk Premium",
    len(risk_df)
)

c3.metric(
    "Average RSI",
    round(
        premium_df["RSI"].mean(),
        2
    )
)

c4.metric(
    "Average Balance",
    round(
        premium_df["Balance"].mean(),
        0
    )
)

st.markdown("---")

left,right = st.columns(2)

with left:

    premium = premium_df[
        "IsActiveMember"
    ].value_counts().reset_index()

    premium.columns=[
        "Status",
        "Count"
    ]

    premium["Status"]=premium[
        "Status"
    ].map({
        1:"Active",
        0:"Inactive"
    })

    fig=px.pie(
        premium,
        names="Status",
        values="Count",
        title="Premium Customer Activity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    fig=px.histogram(
        premium_df,
        x="RSI",
        nbins=20,
        title="Premium Customer RSI"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

risk_chart=risk_df.groupby(
    "Geography"
).size().reset_index()

risk_chart.columns=[
    "Country",
    "Customers"
]

fig=px.bar(
    risk_chart,
    x="Country",
    y="Customers",
    title="At Risk Premium Customers"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")

st.subheader(
    "High Value Customers Requiring Attention"
)

display=risk_df[
    [
        "CustomerId",
        "Geography",
        "Balance",
        "EstimatedSalary",
        "NumOfProducts",
        "RSI"
    ]
]

st.dataframe(
    display,
    use_container_width=True
)

st.markdown("---")

st.success(f"""

### Premium Customer Strategy

• Premium Customers : {len(premium_df)}

• At Risk Premium Customers : {len(risk_df)}

### Suggested Actions

✓ Assign Relationship Manager

✓ Offer Product Bundles

✓ Launch Loyalty Campaign

✓ Personalised Retention Program

✓ Cross Sell Additional Banking Products

""")