import argparse
import pathlib
import pickle

import pandas as pd
from sklearn.svm import SVC

from analysis import Analyser
from maintenance import Maintainer
from model import VehicleInsuranceModel
from preprocess import Preprocessor
from storage import Storage
from validation import Validator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="mlops")
    parser.add_argument(
        "-mode", choices=["inference", "update", "summary"], required=True
    )
    parser.add_argument("-file", type=pathlib.Path)

    args = parser.parse_args()

    if args.mode == "inference":
        df = pd.read_csv(args.file)
        df = Analyser.basic_cleanup(df)
        prep = Preprocessor("inference")
        df = prep.preprocess_nans(df)
        X, _ = prep.preprocess(df)
        model = Maintainer.load_main_model()
        pred = model.predict(X)
        print(pd.DataFrame(pred).value_counts())
    elif args.mode == "update":
        batchsize = 2000  # adjust how much data is added
        storage = Storage(batchsize)
        batch = storage.get_batch()
        print(f"NEW BATCH:\n{batch}")
        storage.load_train_data()
        storage.add_batch(batch)
        storage.save_batch(batch)
        print(f"ORIGINAL DATA WITH NEW BATCH:\n{storage.train_data}")

        Analyser.data_quality(storage.train_data)
        df = Analyser.basic_cleanup(storage.train_data)
        print(f"DATASET AFTER BASIC CLEANUP:\n{df}")

        prep = Preprocessor("train")
        df = prep.preprocess_nans(df)
        print(f"DATASET AFTER NAN REMOVAL:\n{df}")

        X, y = prep.preprocess(df)
        print(f"PREPROCESSED DATA\nX:\n{X}\ny:\n{y}")

        model = VehicleInsuranceModel()
        model.fit(X, y)

        validator = Validator()
        scores = validator.test(model, X, y)
        print(scores)

        Maintainer.save_main_model(model)
    elif args.mode == "summary":
        stats = Analyser.get_stats()
        print("DATA QUALITY:")
        for dict in stats:
            print(dict)
        # print("CROSS VALIDATION SCORES:")
        # model = Maintainer.load_main_model()
        # cv = Validator.cv(model, X, y)
        # print(cv)
