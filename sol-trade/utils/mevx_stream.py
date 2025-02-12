

import requests
import argparse
import json
import time
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from tabulate import tabulate
import csv
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

labelenc = LabelEncoder()
onehot = OneHotEncoder()
minscaler = MinMaxScaler(feature_range=(0, 1))
standscaler = StandardScaler()



def fetch_trades(pool_address, limit, parent, timestamp, trade_id, order_by):
    # Define the API URL with all parameters included
    base_url = f"https://api.mevx.io/api/v1/trades?chain=sol&poolAddress={pool_address}&offset=0&limit={limit}&parent={parent}&timestamp[lte]={timestamp}&usdAmount[gte]=30&orderBy={order_by}"

    # Define headers to mimic a real browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/json",
        "Origin": "https://mevx.io",
        "Referer": "https://mevx.io/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Sec-GPC": "1"
    }

    try:
        # Make the GET request with headers and a timeout
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for HTTP codes 4xx/5xx
        # Parse JSON response
        data = response.json()
        print("Data received:")
        # print(data)
        return data['trades'] if 'trades' in data else []
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def get_pool_address(parent):
    # Define the API URL for fetching pool address
    
    search_url = f"https://api.mevx.io/api/v1/pools/search?q={parent}"

    # Define headers to mimic a real browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/json",
        "Origin": "https://mevx.io",
        "Referer": "https://mevx.io/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Sec-GPC": "1"
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        # print(type(data))
       # Check if 'pools' key exists and contains a list with at least one pool
        if isinstance(data, dict) and 'pools' in data and len(data['pools']) > 0:
            pools_list = data['pools']
            for pool in pools_list:
                # print(pool)
                pool_= pool.get('poolAddress', "")
                # print(pool_)
                # pool_list.append(pool_)
                return pool_
            # If no pool found after looping (shouldn't happen as we checked)
        #  elif i:
        #     print("No pool address found.")
        #     return ""
        else:
            print("Invalid response structure or no pools found.")
            return ""
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch pool address: {e}")
        return ""
    
def monitor_trades(parent_id, limit=50, order_by="timestamp desc", check_interval=0.2, debug=True):
    # Fetch pool address using parent
    pool_address = get_pool_address(parent_id)
    
    buffer_list = []

    if not pool_address:
        print("Failed to fetch pool address. Exiting.")
        return

    prev_trades = []
    while True:
        current_trades = fetch_trades(pool_address, limit, parent_id, int(time.time()), 0, order_by)
        # Compare with previous trades
        if current_trades == prev_trades:
            print("No new trades.") if debug else None
        else:
            # Find new trades by comparing lists
            new_trades = [trade for trade in current_trades if trade not in prev_trades]
            # print(new_trades)
            if new_trades and debug:
                print("\nNew trades found:")
                print(len(new_trades))
                for trade in new_trades:
                    if len(buffer_list) < 50:
                        buffer_list.append(trade)
                        print(f"buffer list {len(buffer_list)}")
                        # print((buffer_list))
                    elif len(buffer_list) > 50:
                        buffer_list.pop()
                        print("condition met!!!")
                        buffer_list.append(trade)

                    if len(buffer_list) == 50:
                        df = pd.DataFrame(buffer_list).sort_values(by='timestamp',  ascending=False).reset_index(drop=True)
                        df = df.drop(columns=["poolAddress", "priceQuote", "quoteAmount", "tokensAmount", "txHash", "metadata", "maker", "id"])
                        df['type'] = labelenc.fit_transform(df['type'])
                        df['token'] = labelenc.fit_transform(df['token'])
                        print(len(buffer_list))
                        buffer_list.pop()
                         # Write the temporary DataFrame to CSV
                        # df.to_csv('buffer_data.csv', index=False)
                        
                        # df_head_ad = df.head()
                        yield (df)
                    else:
                        print(len(buffer_list))
                        print("######### Buffer not yet #########")
                        
        
        # Update previous trades and wait
        prev_trades = current_trades.copy()
        time.sleep(check_interval)



class ServerHandler(BaseHTTPRequestHandler):
    def write_chunk(self, data):
        # Ensure data is bytes
        if isinstance(data, str):
            data = data.encode('utf-8')
        # Calculate chunk size in hexadecimal
        chunk_size = format(len(data), 'x')
        # Write chunk size followed by CRLF, the data, then another CRLF
        self.wfile.write(chunk_size.encode('utf-8'))
        self.wfile.write(b'\r\n')
        self.wfile.write(data)
        self.wfile.write(b'\r\n')

    def do_POST(self):
        if self.path == '/get_trades':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            parent_id = data.get('parent', "")
            limit = data.get('limit', 50)
            order_by = data.get('order_by', "timestamp desc")
            
            try:
                # Fetch trades
                pool_address = get_pool_address(parent_id)
                if not pool_address:
                    self.send_error(400, "Failed to fetch pool address.")
                    return
                
                # Send response headers with chunked encoding
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                
                # for data in monitor_trades(parent_id=parent_id, limit=limit, order_by=order_by):
                #     current_trades = data

                # # Convert DataFrame to JSON
                #     response = current_trades.head(50).to_json(orient='records')
                #     print(f"data before sending to server  {(response)}")
                #     # print(current_trades.head(50))
                #     self.send_response(200)
                #     self.send_header('Content-Type', 'application/json')
                #     self.end_headers()
                #     self.wfile.write(response.encode())
                 # Iterate over the DataFrame chunks generated by monitor_trades
                for df in monitor_trades(parent_id=parent_id, limit=limit, order_by=order_by):
                    # Convert the DataFrame chunk to JSON
                    chunk_data = df.head(50).to_json(orient='records')
                    print(f"Sending chunk: {chunk_data}")
                    # Write the JSON data as a chunk
                    self.write_chunk(chunk_data)
                
                 # Indicate end of chunks
                self.wfile.write(b'0\r\n\r\n')
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404)

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ServerHandler)
    print(f"Server running on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
        print("\nShutting down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch trades data dynamically.")
    parser.add_argument("--pool_address", default="5XH29cos3LeqrTg33xMWjcqSdxmnpe1mKDjrPt7ZcCi8", help="Pool address")
    parser.add_argument("--limit", type=int, default=50, help="Limit of trades")
    parser.add_argument("--parent", default="9toNAaisx9cMPP9f7ees3Ui563pE8W9z4xD32yuXCzmM", help="Parent ID")
    parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp [lte]")
    parser.add_argument("--id", type=int, default=5326854697255068, help="Trade ID [lt]")
    parser.add_argument("--order_by", default="timestamp desc", help="Order by parameter")
    
    args = parser.parse_args()

    # Fetch pool address using parent
    # pool_address = get_pool_address(args.parent)

    # print(pool_address)
    
    # # fetch_trades(pool_address, args.limit, args.parent, args.timestamp, args.id, args.order_by)

    # prev_trades = []
    # while True:
    #     current_trades = fetch_trades(pool_address, args.limit, args.parent, int(time.time()), args.id, args.order_by)
        
    #     # Compare with previous trades
    #     if current_trades == prev_trades:
    #         print("No new trades.")
    #     else:
    #         # Find new trades by comparing lists
    #         new_trades = [trade for trade in current_trades if trade not in prev_trades]
    #         if new_trades:
    #             print("\nNew trades found:")
    #             for trade in new_trades:
    #                 print(json.dumps(trade, indent=2))
        
    #     # Update previous trades and wait for 1 second
    #     prev_trades = current_trades.copy()
    #     time.sleep(0.5)


  # Call the monitoring function with parsed arguments
    # for data in monitor_trades(parent_id=args.parent, limit=args.limit, order_by=args.order_by):
    #     print(data)
    run_server(8000)
        # print(tabulate((data), headers='keys', tablefmt='simple'))
    # except KeyboardInterrupt:
    #     print("\nMonitoring stopped.")