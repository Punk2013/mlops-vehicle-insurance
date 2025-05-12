import pickle
from pathlib import Path

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_validate

from maintenance import Maintainer

models_path = Path("models/")


class Validator:
    def __init__(self):
        self.versions = {}

    @staticmethod
    def test(model, X, y):
        p = model.model.predict_proba(X)[:, 1]
        scores = classification_report(y, p > model.threshold)
        roc_auc = roc_auc_score(y, p)
        print(scores)
        print(roc_auc)

    @staticmethod
    def cv(model, X, y):
        cv = cross_validate(
            model.model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1"]
        )
        for key in cv.keys():
            cv[key] = cv[key].mean()
        return cv

    def add_version(self, model, scores):
        dir_path = Path(models_path)
        max_version = 0

        for file in dir_path.glob("model_ver*.pkl"):
            try:
                version = int(file.stem.split("_ver")[-1])
                if version > max_version:
                    max_version = version
            except (IndexError, ValueError):
                continue
        ver_num = max_version + 1 if max_version > 0 else 1
        self.versions[ver_num] = scores
        Maintainer.save_model_ver(model, ver_num)
