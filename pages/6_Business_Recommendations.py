import streamlit as st

from src.data.loader import DataLoader
from src.features.engagement import EngagementFeature
from src.features.premium_customer import PremiumCustomer
from src.features.relationship_score import RelationshipScore
from src.features.sticky_customer import StickyCustomer

st.set_page_config(
    page_title="Business Recommendations",
    page_icon="💡",
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

st.title("💡 Business Recommendations")

st.markdown("---")

# Cross Sell

cross_sell = df[
    (df["NumOfProducts"] == 1)
    &
    (df["IsActiveMember"] == 1)
]

# Premium Risk

premium_risk = df[
    (df["HighValueCustomer"] == 1)
    &
    (df["IsActiveMember"] == 0)
]

# Sticky

sticky = df[
    df["StickyCustomer"] == 1
]

# High Churn

high_risk = df[
    (df["RSI"] < 50)
    &
    (df["Exited"] == 0)
]

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Cross Sell",
    len(cross_sell)
)

c2.metric(
    "Premium Rescue",
    len(premium_risk)
)

c3.metric(
    "Sticky Reward",
    len(sticky)
)

c4.metric(
    "Retention Target",
    len(high_risk)
)

st.markdown("---")

st.success(f"""

# Strategic Recommendations

## 1 Cross Sell Campaign

Target Customers : {len(cross_sell)}

Action:

• Offer Credit Card

• Offer Investment Products

• Bundle Savings Products

---

## 2 Premium Customer Rescue

Target Customers : {len(premium_risk)}

Action:

• Assign Relationship Manager

• Personalised Banking

• Premium Benefits

---

## 3 Loyalty Program

Target Customers : {len(sticky)}

Action:

• Cashback

• Reward Points

• VIP Banking

---

## 4 Retention Campaign

Target Customers : {len(high_risk)}

Action:

• Engagement Emails

• Mobile Notifications

• Product Recommendations

""")

st.markdown("---")

st.subheader("Cross Sell Customers")

st.dataframe(

    cross_sell[
        [
            "CustomerId",
            "Balance",
            "NumOfProducts",
            "RSI"
        ]
    ],

    use_container_width=True

)

st.markdown("---")

st.subheader("Premium Rescue Customers")

st.dataframe(

    premium_risk[
        [
            "CustomerId",
            "Balance",
            "EstimatedSalary",
            "RSI"
        ]
    ],

    use_container_width=True

)

st.markdown("---")

st.subheader("High Risk Retention Customers")

st.dataframe(

    high_risk[
        [
            "CustomerId",
            "EngagementProfile",
            "RSI",
            "Balance"
        ]
    ],

    use_container_width=True

)