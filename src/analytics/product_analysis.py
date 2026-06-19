class ProductAnalysis:

    @staticmethod
    def churn_by_product(df):

        return (

            df.groupby(
                "NumOfProducts"
            )["Exited"]

            .mean()

            .mul(100)

            .round(2)

        )

    @staticmethod
    def balance_by_product(df):

        return (

            df.groupby(
                "NumOfProducts"
            )["Balance"]

            .mean()

            .round(2)

        )

    @staticmethod
    def salary_by_product(df):

        return (

            df.groupby(
                "NumOfProducts"
            )["EstimatedSalary"]

            .mean()

            .round(2)

        )