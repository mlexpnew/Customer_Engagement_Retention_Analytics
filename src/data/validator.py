class DataValidator:

    REQUIRED_COLUMNS = [
        "CustomerId",
        "Surname",
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited"
    ]

    @staticmethod
    def validate_columns(df):

        missing = []

        for col in DataValidator.REQUIRED_COLUMNS:
            if col not in df.columns:
                missing.append(col)

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        print("Column validation passed.")

    @staticmethod
    def validate_binary(df):

        binary_cols = [
            "HasCrCard",
            "IsActiveMember",
            "Exited"
        ]

        for col in binary_cols:

            values = set(df[col].unique())

            if not values.issubset({0, 1}):
                raise ValueError(
                    f"{col} contains invalid values"
                )

        print("Binary validation passed.")

    @staticmethod
    def validate_missing(df):

        missing = df.isnull().sum().sum()

        if missing > 0:
            print(f"Warning: {missing} missing values")
        else:
            print("No missing values found.")

    @staticmethod
    def validate_duplicates(df):

        dup = df.duplicated().sum()

        print(f"Duplicate Rows : {dup}")