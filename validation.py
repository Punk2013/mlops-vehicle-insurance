import pickle
from pathlib import Path

from sklearn.model_selection import cross_validate

from maintenance import Maintainer

models_path = Path("models/")


class Validator:
    def __init__(self):
        self.versions = {}

    def test(self, model, X, y):
        cv = cross_validate(
            model, X, y, cv=5, scoring=["accuracy", "f1"]
        )  # TODO: add scores
        for key in cv.keys():
            cv[key] = cv[key].mean()
        with open("cache/cv.pkl", "wb") as f:
            pickle.dump(cv, f)
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

