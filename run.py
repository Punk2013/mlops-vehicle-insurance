import argparse
import pickle
import pathlib
import pandas as pd
from storage import Storage
from analysis import Analyser
from preprocess import Preprocessor
from model import VehicleInsuranceModel
from validation import Validator
from maintenance import Maintainer
from sklearn.svm import SVC

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="mlops")
    parser.add_argument("-mode", choices=["inference", "update", "summary"],
                        required=True)
    parser.add_argument("-file", type=pathlib.Path)

    args = parser.parse_args()

    if args.mode == "inference":
        df = pd.read_csv(args.file)
        df = Analyser.basic_cleanup(df)
        prep = Preprocessor('inference')       
        df = prep.preprocess_nans(df)
        X, _ = prep.preprocess(df)
        model = Maintainer.load_main_model()
        pred = model.predict(X)
        print(pred)
    elif args.mode == "update":
        batchsize = 2000 # adjust how much data is added
        storage = Storage(batchsize)
        batch = storage.get_batch()
        print(f"NEW BATCH:\n{batch}")
        storage.load_train_data()
        storage.add_batch(batch)
        print(f"ORIGINAL DATA WITH NEW BATCH:\n{storage.train_data}")

        Analyser.data_quality(storage.train_data)
        df = Analyser.basic_cleanup(storage.train_data)
        print(f"DATASET AFTER BASIC CLEANUP:\n{df}")

        prep = Preprocessor('train')
        df = prep.preprocess_nans(df)
        print(f"DATASET AFTER NAN REMOVAL:\n{df}")
        
        X, y = prep.preprocess(df)
        print(f"PREPROCESSED DATA\nX:\n{X}\ny:\n{y}")

        # model = VehicleInsuranceModel()
        model = SVC()
        model.fit(X, y)

        validator = Validator()
        scores = validator.test(model, X, y)
        print(scores)
        
        Maintainer.save_main_model(model)
        storage.save_train_data()
    elif args.mode == "summary":
        stats = Analyser.get_stats()
        print("DATA QUALITY:")
        for dict in stats:
            print(dict)
        print("CROSS VALIDATION SCORES:")
        with open("cv.pkl", 'rb') as f:
            cv = pickle.load(f)
        print(cv)