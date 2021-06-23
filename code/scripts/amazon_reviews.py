"""
amazon_reviews.py

The purpose of this Python module is 
to take data downloaded from one of the categories at
http://deepyeti.ucsd.edu/jianmo/amazon/index.html,
and eventually output a shuffled dataset 
required for BERT classification. 
"""
import os
import pandas as pd

def load_data(product_category='Appliances', row_limit=None, random_state=42):
    """
    Returns a dataframe containing Amazon reviews.
    Args:
    - product_category (str): type of item purchased
    - random_state (int): seed for repeating experiments
    - row_limit (int): limit on number of rows allowed
    """
    # file_name = f'data\\amazon_reviews\\{product_category}.json.gz' 
    file_name = os.path.join(r'data/amazon_reviews',f'{product_category}.json.gz')
    df = pd.read_json(file_name,compression='infer',lines=True) 

    df["text"] = df["summary"] + ' ' + df["reviewText"]
    df["text"] = df["text"].str.lower()
    
    # df["overall"] = df['overall'].astype(str)

    possible_labels = df.overall.unique()

    label_dict = {possible_label:possible_label - 1 for possible_label in possible_labels}
    # for possible_label in possible_labels:
    #     label_dict[possible_label] = int(possible_label) - 1

    df['label'] = df.overall.replace(label_dict)

    cols = ['text','label']
    df = df[cols]
    df = df.dropna(subset=['label', 'text'])
    
    df = df.sample(frac=1,random_state=random_state) # shuffle

    if row_limit:
        df = df.iloc[:row_limit,:]

    return df

if __name__ == '__main__':
    df = load_data()
    print(df.head())