from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os


class ChurnModel:

    @staticmethod
    def train(X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        os.makedirs("models", exist_ok=True)

        joblib.dump(
            model,
            "models/churn_model.pkl"
        )

        joblib.dump(
            X.columns.tolist(),
            "models/features.pkl"
        )

        return model, accuracy