class ChurnAnalysis:

    @staticmethod
    def overall(df):
        return round(
            df["Exited"].mean() * 100,
            2
        )

    @staticmethod
    def active_vs_inactive(df):

        active = df[
            df["IsActiveMember"] == 1
        ]

        inactive = df[
            df["IsActiveMember"] == 0
        ]

        return {
            "Active":
            round(
                active["Exited"].mean()*100,
                2
            ),

            "Inactive":
            round(
                inactive["Exited"].mean()*100,
                2
            )
        }

    @staticmethod
    def age_churn(df):

        return (

            df.groupby("Age")["Exited"]

            .mean()

            .mul(100)

            .round(2)

        )