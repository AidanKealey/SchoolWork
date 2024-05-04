import gzip
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

path = './review-New_York_10.json'
path_zip = 'review-New_York_10.json.gz'
download = False

if (download):
    def parse(path):
        i = 0
        g = gzip.open(path, 'r')
        for l in g:
            yield json.loads(l)
            i +=1
            print(i)
        print("exit")


            
    data = [json_obj for json_obj in parse(path_zip)]

    # Convert the list into a DataFrame
    df = pd.DataFrame(data)
    # df['time'] = pd.to_datetime(df['time'], unit='ms')
    # df['resp_time'] = pd.to_datetime(df['resp'].apply(lambda x: x.get('time')), unit='ms')
    print(df['rating'].describe())


    df.head()
    df.to_csv('reviews_new_york.csv', index=False)


# Adjust the path to your CSV file
file_path = 'reviews_new_york.csv'

# Read the first 5 rows
df = pd.read_csv(file_path, nrows=5)

# Display the DataFrame
print(df)

df.head()
