import pandas as pd


class Preprocessor:

    @staticmethod
    def prepare(df):

        data = df.copy()

        drop_cols = [
            "CustomerId",
            "Surname",
            "EngagementProfile"
        ]

        for col in drop_cols:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)

        data = pd.get_dummies(
            data,
            columns=["Gender", "Geography"],
            drop_first=True
        )

        X = data.drop("Exited", axis=1)

        y = data["Exited"]

        return X, y