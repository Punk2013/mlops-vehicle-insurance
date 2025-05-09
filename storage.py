from pathlib import Path

import pandas as pd

data_path = Path("data/")
rows_read_path = Path("cache/rows_read.txt")


class Storage:
    def __init__(self, batchsize):
        self.path = data_path / Path("motor_data11-14lats.csv")
        self.train_path = Path("cache/train_data.pkl")

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
        if self.train_path.is_file():
            self.train_data = pd.read_pickle(self.train_path)

    def add_batch(self, batch):
        self.train_data = pd.concat([self.train_data, batch], axis=0, ignore_index=True)

    def save_train_data(self):
        self.train_data.to_pickle(self.train_path)
        with open(rows_read_path, "w") as rows:
            rows.write("%d" % self.train_data.shape[0])

