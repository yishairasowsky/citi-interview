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

from sklearn.model_selection import train_test_split

def load_data(product_category='Appliances', row_limit=None, random_state=42):
    """
    Get shuffled datasets ready for BERT classification. 

    Parameters:
        product_category (str): Type of item purchased.
        random_state (int): Seed for repeating experiments.
        row_limit (int): Maximum number of rows allowed.
        
    Returns:
        df (pd.DataFrame): A table containing training and validation samples.
    """
    # all raw data
    file_name = os.path.join(r'data/amazon_reviews',f'{product_category}.json.gz')
    df = pd.read_json(file_name,compression='infer',lines=True) 

    # select relevant input features
    df["text"] = df["summary"] + ' ' + df["reviewText"]
    df["text"] = df["text"].str.lower()
    
    # organize target variable
    possible_labels = df.overall.unique()
    label_dict = {possible_label:possible_label - 1 for possible_label in possible_labels}
    df['label'] = df.overall.replace(label_dict)

    # select wanted columns
    cols = ['text','label']
    df = df[cols]

    # remove unwanted rows
    df = df.dropna(subset=['label', 'text'])
    
    # shuffle rows
    df = df.sample(frac=1,random_state=random_state) 

    if row_limit:
        df = df.iloc[:row_limit,:]

    # split samples into training versus validation
    X_train, X_val, _, _ = train_test_split(df.index.values, 
                                                    df.label.values, 
                                                    test_size=0.15, 
                                                    random_state=42, 
                                                    stratify=df.label.values)
    # label data type in column
    df['data_type'] = ['not_set']*df.shape[0] 
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'

    return df

if __name__ == '__main__':
    df = load_data()
    print(df.head())