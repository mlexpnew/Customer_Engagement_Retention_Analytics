import joblib
import pandas as pd


class Predictor:

    def __init__(self):

        self.model = joblib.load(
            "models/churn_model.pkl"
        )

        self.features = joblib.load(
            "models/features.pkl"
        )

    def predict(self, data):

        df = pd.DataFrame(
            [data]
        )

        df = df.reindex(
            columns=self.features,
            fill_value=0
        )

        probability = self.model.predict_proba(df)[0][1]

        prediction = self.model.predict(df)[0]

        return prediction, probability