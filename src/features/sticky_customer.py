class StickyCustomer:

    @staticmethod
    def create(df):

        df["StickyCustomer"] = (

            (df["IsActiveMember"] == 1)

            &

            (df["NumOfProducts"] >= 2)

            &

            (df["HasCrCard"] == 1)

            &

            (df["Tenure"] >= 5)

            &

            (df["RSI"] >= 70)

        ).astype(int)

        return df