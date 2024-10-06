import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import torch

def preprocess_data(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file, delimiter=',', on_bad_lines='warn')
    
    # Check for missing values and handle them if necessary
    if df.isnull().any().any():
        print("Warning: Missing values found in dataset.")
        df.fillna(method='ffill', inplace=True)  # Example of handling missing values, forward fill
    
    # Drop the columns 'id' and 'blockHash'
    df.drop(['id', 'blockHash'], axis=1, inplace=True)
    
    # Convert the 'maker' column to numeric identifiers if it is not already numeric
    if df['maker'].dtype == object:  # Check if column is of type object (non-numeric)
        label_encoder = LabelEncoder()
        df['maker'] = label_encoder.fit_transform(df['maker'])
    
    # One-hot encode the 'type' column
    ohe = OneHotEncoder()
    type_encoded = ohe.fit_transform(df[['type']]).toarray()
    df_encoded = pd.DataFrame(type_encoded, columns=['type_sell', 'type_buy'])
    
    # Concatenate the encoded features with the original DataFrame (excluding the original 'type' column)
    df = pd.concat([df, df_encoded], axis=1).drop('type', axis=1)
    
    # Standardize numerical features
    scaler = StandardScaler()
    num_features = ['amountToken', 'amountETH', 'amountRef', 'price', 'pool_liquidity']
    df[num_features] = scaler.fit_transform(df[num_features])

    # Ensure all columns are numeric before converting to PyTorch tensor
    for col in num_features + ['type_sell', 'type_buy']:  # Include the one-hot encoded columns
        if df[col].dtype == object:  # Check if column is of type object (non-numeric)
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN


    print(df.head(90))


    # Now convert the entire DataFrame to a PyTorch tensor
    tensor_data = torch.tensor(df.values, dtype=torch.float32)
    
    return tensor_data

# Example usage:
csv_file_path = '/home/abishek/sol-proj/ray/sol-trade/output.csv'
data = preprocess_data(csv_file_path)
print(data)