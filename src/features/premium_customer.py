class PremiumCustomer:

    @staticmethod
    def create(df):

        balance_limit = df["Balance"].quantile(0.75)

        salary_limit = df["EstimatedSalary"].quantile(0.75)

        df["HighValueCustomer"] = (

            (df["Balance"] >= balance_limit)

            |

            (df["EstimatedSalary"] >= salary_limit)

        ).astype(int)

        return df