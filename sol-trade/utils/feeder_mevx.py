import numpy as np
import pandas as pd
import random
import collections
from typing import Dict
from typing import Optional
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
import pickle
from tabulate import tabulate

import os
from dotenv import load_dotenv

load_dotenv()



# from ray.tune.registry import get_trainable_cls, register_env  # noqa
# from ray import data,remote

RAY = False

class DataLoaderMevx:
    def __init__(self, csv_file_path: Optional[str] = None, 
                 pckl_file_path: Optional[str] = None, 
                 df : pd.DataFrame = None, print_tabulate=False):
        

        assert isinstance(df, pd.DataFrame), "df_file must be a pandas DataFrame"

        if pckl_file_path:
            with open(pckl_file_path, 'rb') as f:
                self.df = pickle.load(f)

        elif csv_file_path:
            self.df = pd.read_csv(csv_file_path)

        elif isinstance(df, pd.DataFrame):
            # if df is not None:
            #     assert isinstance(df, pd.DataFrame), "df_file must be a pandas DataFrame"
            self.df = df
        else:
            raise ValueError("Either CSV or Pickle path must be provided.")


        if RAY:
            self.df = self.df.drop(columns=["poolAddress", "priceQuote", "quoteAmount", "tokensAmount", "txHash", "maker"])
        else:
            self.df = self.df.drop(columns=["poolAddress", "priceQuote", "quoteAmount", "tokensAmount", "txHash", "metadata", "maker"])

        self.df_head_ad = self.df.head()

        if print_tabulate:
            print("First 5 rows of the DataFrame:")
            print(tabulate(self.df.head(11), headers='keys', tablefmt='grid'))

        # self.ds = data.from_pandas(self.df)
        self.labelenc = LabelEncoder()
        self.onehot = OneHotEncoder()
        self.data_que = collections.deque()
        self.minscaler = MinMaxScaler(feature_range=(0, 1))
        self.standscaler = StandardScaler()

    def label_encode(self,batch):
        # batch["maker"] = np.clip(batch["maker"] + 4, 0, 255)
        batch["maker"] = self.labelenc.fit_transform(batch["maker"])
        return batch


    def ray_load(self, eval):
        # ds = data.read_csv(csv_file_path)
        # integer_encoded = self.labelenc.fit_transform(data)
        #self.ds = self.ds.map_batches(self.label_encode)
        #batch = self.ds.take_batch(batch_size=40, batch_format="pandas")
        self.df['type'] = self.labelenc.fit_transform(self.df['type'])
        self.df['token'] = self.labelenc.fit_transform(self.df['token'])
        # self.df['maker'] = self.labelenc.fit_transform(self.df['maker']) #remove maker
        # print(self.df.head(1))
        self.df.columns = self.df.iloc[0]
        self.df = self.df.iloc[1:]
        # Reverse the order of rows
        # if eval:
        #     self.df = self.df.iloc[::-1].reset_index(drop=True)
        # print(self.df.head(1))
        return self.df

    def enqueue(self, data_item):
        """Enqueues a single piece of data into the queue."""
        if isinstance(data_item, list):
            for item in data_item:
                self.data_que.append(item)


    def dequeue(self):
        """Dequeues and returns the oldest piece of data from the queue."""
        if not self.data_que:
            raise IndexError("The queue is empty.")
        return self.data_que.popleft()

    def normalize(self, queue):
        """Normalizes the data in the queue."""
        normalized_data = self.standscaler(queue)
        return normalized_data

    def manage_que(self,item):
        """Manages the queue by adding new data and removing old data if necessary."""
        if len(self.data_que) <= 50:
            self.enqueue(item)
        elif len(self.data_que) > 50:
            self.dequeue()
            self.enqueue(item)

        return self.normalize(self.data_que)


    def take(self, num: int):
        return self.ds.take(num)



# Example usage:
# csv_file_path = os.getenv("CSV_FILE")
# data_loader = DataLoaderMevx(csv_file_path)
# # print(data_loader.df_head_bd)
# # print(data_loader.df_head_ad)
# # print(data_loader.ds.take_batch(1))
# ds = data_loader.ray_load()
#batch = ds.take_batch(batch_size=10, batch_format="pandas")
# ds.show(1)