class GeographyAnalysis:

    @staticmethod
    def churn(df):

        return (

            df.groupby(
                "Geography"
            )["Exited"]

            .mean()

            .mul(100)

            .round(2)

        )

    @staticmethod
    def customers(df):

        return (

            df[
                "Geography"
            ]

            .value_counts()

        )

    @staticmethod
    def average_balance(df):

        return (

            df.groupby(
                "Geography"
            )["Balance"]

            .mean()

            .round(2)

        )