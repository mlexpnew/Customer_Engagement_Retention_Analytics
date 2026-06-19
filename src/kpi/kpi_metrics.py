import pandas as pd


class KPIMetrics:

    @staticmethod
    def overall_churn(df):

        return round(
            df["Exited"].mean() * 100,
            2
        )

    @staticmethod
    def active_churn(df):

        active = df[
            df["IsActiveMember"] == 1
        ]

        inactive = df[
            df["IsActiveMember"] == 0
        ]

        return {

            "Active Churn %":
            round(active["Exited"].mean()*100, 2),

            "Inactive Churn %":
            round(inactive["Exited"].mean()*100, 2)

        }

    @staticmethod
    def geography_churn(df):

        return (

            df.groupby(
                "Geography"
            )["Exited"]

            .mean()

            .mul(100)

            .round(2)

        )

    @staticmethod
    def gender_churn(df):

        return (

            df.groupby(
                "Gender"
            )["Exited"]

            .mean()

            .mul(100)

            .round(2)

        )

    @staticmethod
    def product_churn(df):

        return (

            df.groupby(
                "NumOfProducts"
            )["Exited"]

            .mean()

            .mul(100)

            .round(2)

        )

    @staticmethod
    def premium_churn(df):

        premium = df[
            df["HighValueCustomer"] == 1
        ]

        normal = df[
            df["HighValueCustomer"] == 0
        ]

        return {

            "Premium":
            round(
                premium["Exited"].mean()*100,
                2
            ),

            "Regular":
            round(
                normal["Exited"].mean()*100,
                2
            )

        }

    @staticmethod
    def sticky_rate(df):

        return round(

            df["StickyCustomer"].mean()

            * 100,

            2

        )

    @staticmethod
    def average_rsi(df):

        return round(

            df["RSI"].mean(),

            2

        )

    @staticmethod
    def engagement_distribution(df):

        return (

            df[
                "EngagementProfile"
            ]

            .value_counts()

        )