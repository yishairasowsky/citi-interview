import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen

### load the meta data
data = []
with gzip.open('meta_Computers.json.gz') as f:
    for l in f:
        data.append(json.loads(l.strip()))
    
# total length of list, this number equals total number of products
print(len(data))

# first row of the list
print(data[0])

# convert list into pandas dataframe

df = pd.DataFrame.from_dict(data)

print(len(df))

### remove rows with unformatted title (i.e. some 'title' may still contain html style content)

df3 = df.fillna('')
df4 = df3[df3.title.str.contains('getTime')] # unformatted rows
df5 = df3[~df3.title.str.contains('getTime')] # filter those unformatted rows
print(len(df4))
print(len(df5))

# how those unformatted rows look like
df4.iloc[0]

df.head()

# !jupyter nbconvert --to script parse_and_clean_meta_data.ipynb
!jupyter nbconvert --to script *.ipynb



from google.colab import drive
drive.mount('/content/drive')

