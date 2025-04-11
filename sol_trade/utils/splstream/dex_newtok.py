import requests
import json

url = f"https://api.dexscreener.com/token-boosts/latest/v1"



response = requests.get(url)
print(type(response))
data = response.json()
# print(data)
for item in data:
    if item["chainId"] == "solana":
        # print(item["tokenAddress"])
        token = item["tokenAddress"]
        url2 = f"https://api.dexscreener.com/tokens/v1/solana/{token}"
        url3 = f"https://api.dexscreener.com/token-pairs/v1/solana/{token}"

        resp2 = requests.get(url2)
        resp3 = requests.get(url3)
        data2 = resp2.json() 
        data3 = resp3.json() 

        for item2 in data2:
            print(item2["volume"])
            print(item2["fdv"])
        
        for item3 in data3:
            print(item3["dexId"])
            if item3["dexId"] == "moonshot":
                print(item3["moonshot"])
        
        
        
            

