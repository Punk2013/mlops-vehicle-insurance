import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
from pathlib import Path

cat_cols = ['SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE']
num_cols = ['INSR_BEGIN', 'INSR_END', 'EFFECTIVE_YR', 'INSURED_VALUE', 'PREMIUM', 'PROD_YEAR', 'SEATS_NUM', 'CARRYING_CAPACITY', 'CCM_TON']
num_cols_new = ['BEGIN_YEAR', 'BEGIN_MONTH', 'BEGIN_DAY', 'END_YEAR', 'END_MONTH', 'END_DAY',
            'EFFECTIVE_YR', 'INSURED_VALUE', 'PREMIUM', 'PROD_YEAR', 'SEATS_NUM', 'CARRYING_CAPACITY', 'CCM_TON']

transformer_path = Path("transformer.pkl")

class Preprocessor:
    def __init__(self, mode):
        self.transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), cat_cols),
                ('num', StandardScaler(), num_cols_new)
            ]
        ) 

        assert mode == 'train' or mode == 'inference', f"Unknown mode: {mode}"
        self.mode = mode

    @staticmethod
    def preprocess_nans(df):
        for col in cat_cols:
            if col == 'TYPE_VEHICLE':
                df[col] = df[col].fillna(df[col].mode()[0])
                continue
            if df[col].isna().sum() / df.shape[0] < 0.05:
                df.dropna(subset=col, inplace=True)
            else:
                df[col] = df.groupby('TYPE_VEHICLE')[col].transform(
                    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
            )
        
        for col in num_cols:
            if col == 'CARRYING_CAPACITY':
                df[col] = df[col].fillna(0)
                continue
            if df[col].isna().sum() / df.shape[0] < 0.05:
                df.dropna(subset=col, inplace=True)
            else:
                df[col] = df.groupby('TYPE_VEHICLE')[col].transform(
                    lambda x: x.fillna(x.median()) if x.notnan().any() else 0)
        return df

    def preprocess(self, df):
        insr_begin = pd.to_datetime(df['INSR_BEGIN'], errors='coerce', format="%d-%b-%y")
        df['BEGIN_YEAR'] = insr_begin.dt.year
        df['BEGIN_MONTH'] = insr_begin.dt.month
        df['BEGIN_DAY'] = insr_begin.dt.day
        df = df.drop('INSR_BEGIN', axis=1)

        insr_end = pd.to_datetime(df['INSR_END'], errors='coerce', format="%d-%b-%y")
        df['END_YEAR'] = insr_end.dt.year
        df['END_MONTH'] = insr_end.dt.month
        df['END_DAY'] = insr_end.dt.day
        df = df.drop('INSR_END', axis=1)

        df['EFFECTIVE_YR'] = pd.to_numeric(df['EFFECTIVE_YR'], errors='coerce')
        if df['EFFECTIVE_YR'].isna().sum() / df.shape[0] < 0.05:
                df.dropna(subset='EFFECTIVE_YR', inplace=True)
        else:
            df['EFFECTIVE_YR'].fillna(0, inplace=True)

        for col in num_cols_new:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        X = df.drop('CLAIM_PAID', axis=1)
        y = df['CLAIM_PAID'].fillna(0)
        y = (y!=0)

        if transformer_path.is_file():
            with open("transformer.pkl", 'rb') as f:
                self.transformer = pickle.load(f)

        if self.mode == 'train':
            self.transformer.fit(X)
            with open("transformer.pkl", 'wb') as f:
                pickle.dump(self.transformer, f)
        X = self.transformer.transform(X)

        return X, y