import numpy as np


class EngagementFeature:

    @staticmethod
    def create(df):

        balance_median = df["Balance"].median()

        conditions = [

            (df["IsActiveMember"] == 1) &
            (df["NumOfProducts"] >= 2),

            (df["IsActiveMember"] == 0) &
            (df["Balance"] > balance_median),

            (df["IsActiveMember"] == 1),

            (df["IsActiveMember"] == 0)

        ]

        choices = [

            "Highly Engaged",

            "Inactive High Balance",

            "Active Low Product",

            "Disengaged"

        ]

        df["EngagementProfile"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )

        return df