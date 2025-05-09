import pickle
from pathlib import Path

models_path = Path("models/")


class Maintainer:
    @staticmethod
    def save_main_model(model):
        with open(models_path / Path("main_model.pkl"), "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_main_model():
        with open(models_path / Path("main_model.pkl"), "rb") as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def save_model_ver(model, ver_num):
        with open(models_path / Path(f"model_ver{ver_num}.pkl"), "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model_ver(ver_num):
        with open(models_path / Path(f"model_ver{ver_num}.pkl"), "rb") as f:
            model = pickle.load(f)
        return model
