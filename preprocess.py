import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

cat_cols = ["SEX", "INSR_TYPE", "TYPE_VEHICLE", "USAGE"]
num_cols = [
    "INSR_BEGIN",
    "INSR_END",
    "EFFECTIVE_YR",
    "INSURED_VALUE",
    "PREMIUM",
    "PROD_YEAR",
    "SEATS_NUM",
    "CARRYING_CAPACITY",
    "CCM_TON",
]
num_cols_new = [
    "BEGIN_YEAR",
    "BEGIN_MONTH",
    "BEGIN_DAY",
    "END_YEAR",
    "END_MONTH",
    "END_DAY",
    "EFFECTIVE_YR",
    "INSURED_VALUE",
    "PREMIUM",
    "PROD_YEAR",
    "SEATS_NUM",
    "CARRYING_CAPACITY",
    "CCM_TON",
]

transformer_path = Path("cache/transformer.pkl")


class Preprocessor:
    def __init__(self, mode):
        self.transformer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols),
                ("num", StandardScaler(), num_cols_new),
            ]
        )

        assert mode == "train" or mode == "inference", f"Unknown mode: {mode}"
        self.mode = mode

    @staticmethod
    def preprocess_nans(df):
        for col in cat_cols:
            if col == "TYPE_VEHICLE":
                df[col] = df[col].fillna(df[col].mode()[0])
                continue
            if df[col].isna().sum() / df.shape[0] < 0.05:
                df = df.dropna(subset=col)
            else:
                df[col] = df.groupby("TYPE_VEHICLE")[col].transform(
                    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
                )

        for col in num_cols:
            if col == "CARRYING_CAPACITY":
                df[col] = df[col].fillna(0)
                continue
            if df[col].isna().sum() / df.shape[0] < 0.05:
                df = df.dropna(subset=col)
            else:
                df[col] = df.groupby("TYPE_VEHICLE")[col].transform(
                    lambda x: x.fillna(x.median()) if x.notnan().any() else 0
                )
        return df

    def preprocess(self, df):
        df = df.drop("OBJECT_ID", axis=1)
        insr_begin = pd.to_datetime(
            df["INSR_BEGIN"], errors="coerce", format="%d-%b-%y"
        )
        df["BEGIN_YEAR"] = insr_begin.dt.year.astype('float')
        df["BEGIN_MONTH"] = insr_begin.dt.month.astype('float')
        df["BEGIN_DAY"] = insr_begin.dt.day.astype('float')
        df = df.drop("INSR_BEGIN", axis=1)

        insr_end = pd.to_datetime(df["INSR_END"], errors="coerce", format="%d-%b-%y")
        df["END_YEAR"] = insr_end.dt.year.astype('float')
        df["END_MONTH"] = insr_end.dt.month.astype('float')
        df["END_DAY"] = insr_end.dt.day.astype('float')
        df = df.drop("INSR_END", axis=1)

        df["EFFECTIVE_YR"] = pd.to_numeric(df["EFFECTIVE_YR"], errors="coerce")
        if df["EFFECTIVE_YR"].isna().sum() / df.shape[0] < 0.05:
            df = df.dropna(subset="EFFECTIVE_YR")
        else:
            df["EFFECTIVE_YR"].fillna(0, inplace=True)

        for col in num_cols_new:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        # df.loc[:, cat_cols] = df[cat_cols].astype(object)

        X = df.drop("CLAIM_PAID", axis=1)
        y = df["CLAIM_PAID"].notnull()

        if transformer_path.is_file():
            with open("cache/transformer.pkl", "rb") as f:
                self.transformer = pickle.load(f)

        if self.mode == "train":
            self.transformer.fit(X)
            with open("cache/transformer.pkl", "wb") as f:
                pickle.dump(self.transformer, f)
        print(X.info())
        X = self.transformer.transform(X)

        return X, y
