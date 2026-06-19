import streamlit as st
import plotly.express as px

from src.data.loader import DataLoader
from src.features.engagement import EngagementFeature
from src.features.premium_customer import PremiumCustomer
from src.features.relationship_score import RelationshipScore
from src.features.sticky_customer import StickyCustomer

st.set_page_config(
    page_title="Product Analytics",
    page_icon="💳",
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

st.title("💳 Product Analytics")

st.markdown("---")

# KPIs

c1,c2,c3,c4 = st.columns(4)

avg_products = round(
    df["NumOfProducts"].mean(),
    2
)

single = round(
    (df["NumOfProducts"] == 1).mean()*100,
    2
)

multi = round(
    (df["NumOfProducts"] >= 2).mean()*100,
    2
)

avg_balance = round(
    df["Balance"].mean(),
    2
)

c1.metric(
    "Avg Products",
    avg_products
)

c2.metric(
    "Single Product",
    f"{single}%"
)

c3.metric(
    "Multi Product",
    f"{multi}%"
)

c4.metric(
    "Avg Balance",
    f"{avg_balance:,.0f}"
)

st.markdown("---")

left,right = st.columns(2)

with left:

    churn = df.groupby(
        "NumOfProducts"
    )["Exited"].mean().reset_index()

    churn["Exited"] *= 100

    fig = px.bar(
        churn,
        x="NumOfProducts",
        y="Exited",
        title="Product Count vs Churn"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    balance = df.groupby(
        "NumOfProducts"
    )["Balance"].mean().reset_index()

    fig = px.bar(
        balance,
        x="NumOfProducts",
        y="Balance",
        title="Average Balance by Product"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

left,right = st.columns(2)

with left:

    salary = df.groupby(
        "NumOfProducts"
    )["EstimatedSalary"].mean().reset_index()

    fig = px.line(
        salary,
        x="NumOfProducts",
        y="EstimatedSalary",
        markers=True,
        title="Salary by Product"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    products = df["NumOfProducts"].value_counts().reset_index()

    products.columns = [
        "Products",
        "Customers"
    ]

    fig = px.pie(
        products,
        names="Products",
        values="Customers",
        title="Product Mix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

st.subheader("Cross-Sell Opportunity")

cross_sell = df[
    (df["NumOfProducts"] == 1)
    &
    (df["IsActiveMember"] == 1)
]

st.success(f"""

### Business Recommendations

• Customers with one product : {len(df[df['NumOfProducts']==1])}

• Active one-product customers : {len(cross_sell)}

• Multi-product customers : {len(df[df['NumOfProducts']>=2])}

• Cross-sell target segment : {len(cross_sell)}

""")

st.markdown("---")

with st.expander("View Product Data"):

    st.dataframe(
        df[
            [
                "CustomerId",
                "NumOfProducts",
                "Balance",
                "EstimatedSalary",
                "IsActiveMember",
                "Exited"
            ]
        ],
        use_container_width=True
    )