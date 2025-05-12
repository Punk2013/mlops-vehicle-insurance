from pathlib import Path

from catboost import CatBoostClassifier

models_path = Path("models/")


class VehicleInsuranceModel:
    def __init__(self):
        self.model = CatBoostClassifier(auto_class_weights="Balanced")
        self.threshold = 0.7

    def fit(self, X, y):
        self.model.fit(X, y)

    def partial_fit(self, X, y, model=models_path / Path("main_model.pkl")):
        self.model.fit(X, y, init_model=model)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1] > self.threshold

    def get_params(self, deep=True):
        return self.model.get_params()
