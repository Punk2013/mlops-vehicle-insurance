import re
from pathlib import Path

import pandas as pd

data_path = Path("data/")
rows_read_path = Path("cache/rows_read.txt")


class Storage:
    def __init__(self, batchsize):
        self.path = data_path / Path("motor_data11-14lats.csv")
        self.train_path = Path("cache/train_data/")

        self.train_data = pd.DataFrame()
        self.batchsize = batchsize

        if rows_read_path.is_file():
            with open(rows_read_path, "r") as rows:
                self.reader = pd.read_csv(
                    self.path,
                    chunksize=self.batchsize,
                    skiprows=range(1, int(rows.read())),
                )
        else:
            self.reader = pd.read_csv(self.path, chunksize=self.batchsize)

    def get_batch(self):
        return next(self.reader)

    def load_train_data(self):
        batch_files = list(self.train_path.glob("batch_*.pkl"))
        if not batch_files:
            return

        def extract_startrow(file: Path) -> int:
            match = re.search(r"batch_(\d+)-(\d+)\.pkl", file.name)
            if not match:
                raise ValueError(f"Invalid filename format: {file.name}")
            return int(match.group(1))

        batch_files_sorted = sorted(batch_files, key=extract_startrow)
        self.train_data = pd.concat(
            [pd.read_pickle(f) for f in batch_files_sorted], ignore_index=True
        )

    def add_batch(self, batch):
        self.train_data = pd.concat([self.train_data, batch], axis=0, ignore_index=True)

    def save_batch(self, batch):
        read = 0
        if rows_read_path.is_file():
            with open(rows_read_path, "r") as rows:
                read = int(rows.read())

        batch.to_pickle(
            self.train_path / Path(f"batch_{read + 1}-{read + self.batchsize}.pkl")
        )

        with open(rows_read_path, "w") as rows:
            rows.write(f"{read + self.batchsize}")

    @staticmethod
    def calc_metaparams(self, batch):
        batch_mean = {}
        batch_mean["INSURED_VALUE"] = pd.mean(batch["INSURED_VALUE"])
        batch_mean["PREMIUM"] = pd.mean(batch["PREMIUM"])
        batch_mean["SEATS_NUM"] = pd.mean(batch["SEATS_NUM"])
        batch_mean["CARRYING_CAPACITY"] = pd.mean(batch["CARRYING_CAPACITY"])
        batch_mean["CCM_TON"] = pd.mean(batch["CCM_TON"])
        batch_mean["CLAIM_PAID"] = pd.mean(batch["CLAIM_PAID"])
        batch_std = {}
        batch_std["INSURED_VALUE"] = pd.std(batch["INSURED_VALUE"])
        batch_std["PREMIUM"] = pd.std(batch["PREMIUM"])
        batch_std["SEATS_NUM"] = pd.std(batch["SEATS_NUM"])
        batch_std["CARRYING_CAPACITY"] = pd.std(batch["CARRYING_CAPACITY"])
        batch_std["CCM_TON"] = pd.std(batch["CCM_TON"])
        batch_std["CLAIM_PAID"] = pd.std(batch["CLAIM_PAID"])
        return batch_mean, batch_std
