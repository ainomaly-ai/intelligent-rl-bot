import numpy as np
import pandas as pd
import random



from typing import Optional

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env  # noqa
from ray import data 

csv_file_path = "/home/abishek/sol-proj/ray/sol-trade/output.csv"

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)

df = df.drop(columns= [ "id","blockNumber","blockHash", "logIndex", "amountETH","priceETH", "pool_amountToken", "pool_amountTokenRef","amountRef"])


# Convert the DataFrame to a Ray dataset
ds = data.from_pandas(df)


print(ds.take(1))