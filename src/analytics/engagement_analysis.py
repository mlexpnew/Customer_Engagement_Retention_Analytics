
class EngagementAnalysis:

    @staticmethod
    def profile_distribution(df):

        return (

            df[
                "EngagementProfile"
            ]

            .value_counts()

        )

    @staticmethod
    def profile_churn(df):

        return (

            df.groupby(
                "EngagementProfile"
            )["Exited"]

            .mean()

            .mul(100)

            .round(2)

        )

    @staticmethod
    def sticky_customers(df):

        return (

            df[
                "StickyCustomer"
            ]

            .value_counts()

        )