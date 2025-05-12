import pickle
from datetime import datetime

import numpy as np
import pandas as pd


class Analyser:
    @staticmethod
    def basic_cleanup(df):
        df = df.drop_duplicates()
        df.loc[:, "SEX"] = df["SEX"].where(df["SEX"].isin([0, 1, 2]), np.nan)
        df.loc[:, "INSR_TYPE"] = df["INSR_TYPE"].where(
            df["INSR_TYPE"].isin([1201, 1202, 1204]), np.nan
        )
        df.loc[:, "TYPE_VEHICLE"] = df["TYPE_VEHICLE"].where(
            df["TYPE_VEHICLE"].isin(
                [
                    "Pick-up",
                    "Station Wagones",
                    "Truck",
                    "Bus",
                    "Automobile",
                    "Tanker",
                    "Trailers and semitrailers",
                    "Motor-cycle",
                    "Tractor",
                    "Special construction",
                    "Trade plates",
                ]
            ),
            np.nan,
        )
        df.loc[:, "USAGE"] = df["USAGE"].where(
            df["USAGE"].isin(
                [
                    "Own Goods",
                    "Private",
                    "General Cartage",
                    "Fare Paying Passengers",
                    "Taxi",
                    "Car Hires",
                    "Own service",
                    "Agricultural Own Farm",
                    "Special Construction",
                    "Others",
                    "Learnes",
                    "Ambulance",
                    "Agricultural Any Farm",
                    "Fire fighting",
                ]
            ),
            np.nan,
        )

        for p in [
            "PREMIUM",
            "INSURED_VALUE",
            "CLAIM_PAID",
            "SEATS_NUM",
            "CARRYING_CAPACITY",
            "CCM_TON",
        ]:
            df.loc[:, p] = df[p].where(df[p] >= 0, np.nan)

        today = datetime.now()
        for p in ["INSR_BEGIN", "INSR_END"]:
            date = pd.to_datetime(df[p], errors="coerce", format="%d-%b-%y")
            df.loc[:, p] = df[p].where((date.dt.year >= 1886) & (date <= today), np.nan)

        df.loc[:, "PROD_YEAR"] = df["PROD_YEAR"].where(
            (df["PROD_YEAR"] >= 1886) & (df["PROD_YEAR"] <= today.year), np.nan
        )
        return df

    @staticmethod
    def data_quality(df):
        # completeness
        df_bin = df.isna()
        cols_nan_ratio = df_bin.sum(axis=0) / df.shape[0]
        rows_nan_ratio = df_bin.sum(axis=1) / df.shape[1]
        completeness = {
            "full": df_bin.sum().sum() / np.prod(df.shape),
            "cols_max": cols_nan_ratio.max(),
            "rows_max": rows_nan_ratio.max(),
        }

        # validity
        validity = {}
        for p in [
            "PREMIUM",
            "INSURED_VALUE",
            "CLAIM_PAID",
            "SEATS_NUM",
            "CARRYING_CAPACITY",
            "CCM_TON",
        ]:
            validity[p] = not ((df[p] < 0).any())

        # timeliness
        timeliness = {}
        max_time_lap = np.diff(
            pd.to_datetime(
                df["INSR_BEGIN"], errors="coerce", format="%d-%b-%y"
            ).unique()
        ).max()
        max_time_lap = max_time_lap.astype("timedelta64[D]").astype(int)
        timeliness["delta_time_max"] = max_time_lap

        # save to file
        with open("cache/data_quality.pkl", "wb") as f:
            pickle.dump((completeness, validity, timeliness), f)

    @staticmethod
    def get_stats():
        with open("cache/data_quality.pkl", "rb") as f:
            stats = pickle.load(f)
        return stats
