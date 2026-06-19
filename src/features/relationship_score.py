class RelationshipScore:

    @staticmethod
    def normalize(series):

        if series.max() == series.min():
            return series * 0

        return (

            series - series.min()

        ) / (

            series.max() - series.min()

        )

    @staticmethod
    def create(df):

        activity = df["IsActiveMember"]

        products = RelationshipScore.normalize(
            df["NumOfProducts"]
        )

        tenure = RelationshipScore.normalize(
            df["Tenure"]
        )

        card = df["HasCrCard"]

        credit = RelationshipScore.normalize(
            df["CreditScore"]
        )

        score = (

            0.40 * activity +

            0.25 * products +

            0.15 * tenure +

            0.10 * card +

            0.10 * credit

        )

        df["RSI"] = (score * 100).round(2)

        return df