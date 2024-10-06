import numpy as np
import pandas as pd
import random
import collections
from typing import Dict
from typing import Optional
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler

from ray.tune.registry import get_trainable_cls, register_env  # noqa
from ray import data,remote 

class DataLoader:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(csv_file_path)
        self.df = self.df.drop(columns=["id", "blockNumber", "blockHash", "logIndex", "amountETH", "priceETH", "pool_amountToken", "pool_amountTokenRef", "amountRef"])
        self.ds = data.from_pandas(self.df)
        self.labelenc = LabelEncoder()
        self.onehot = OneHotEncoder()
        self.data_que = collections.deque()        
        self.minscaler = MinMaxScaler(feature_range=(0, 1))
        self.standscaler = StandardScaler()

    def label_encode(self,batch):
        # batch["maker"] = np.clip(batch["maker"] + 4, 0, 255)
        batch["maker"] = self.labelenc.fit_transform(batch["maker"])
        return batch
    

    def ray_load(self):
        # ds = data.read_csv(csv_file_path)
        # integer_encoded = self.labelenc.fit_transform(data)
        self.ds = self.ds.map_batches(self.label_encode)
        batch = self.ds.take_batch(batch_size=40, batch_format="pandas")
        print(self.ds.schema())
        print(self.ds.take(1))
        return self.ds
    
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
    
    # Ray DataLoader function
    @remote
    def create_data_loader(data_size, batch_size):
        dataset = ExampleDataset(data_size)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            


    def take(self, num: int):
        return self.ds.take(num)
    
    def observation_data(self):
        self.timestamp = x
        self.maker = d
        self.token = e
        self.current_price = ds        


# Example usage:
csv_file_path = "/home/abishek/sol-proj/ray/sol-trade/output.csv"
data_loader = DataLoader(csv_file_path)
# print(data_loader.ds.take_batch(1))
ds = data_loader.ray_load()
#batch = ds.take_batch(batch_size=10, batch_format="pandas")
# ds.show(1)