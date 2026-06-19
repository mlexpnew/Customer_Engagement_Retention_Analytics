import pandas as pd
from pathlib import Path


class DataLoader:

    def __init__(self, file_path="data/European_Bank.xlsx"):
        self.file_path = Path(file_path)

    def load_data(self):

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        try:
            df = pd.read_excel(self.file_path)

            print("=" * 50)
            print("Dataset Loaded Successfully")
            print("=" * 50)
            print(f"Rows    : {df.shape[0]}")
            print(f"Columns : {df.shape[1]}")
            print("=" * 50)

            return df

        except Exception as e:
            raise Exception(
                f"Error loading dataset: {e}"
            )

    def dataset_info(self):

        df = self.load_data()

        print(df.info())

        print("\nMissing Values")
        print(df.isnull().sum())

        print("\nDuplicate Rows")
        print(df.duplicated().sum())

        print("\nFirst Five Rows")
        print(df.head())