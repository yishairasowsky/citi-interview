"""
*************
kaggle_ner.py
*************
The purpose of this Python module is 
to take data downloaded from one of the categories at
https://www.kaggle.com/alaakhaled/conll003-englishversion,
and eventually output a shuffled dataset required for BERT classification. 
"""
import pandas as pd

def load_data(type='train',random_state=42):
    """
    Returns a dataframe containing Kaggle's NER labeled entities.
    Args:
    - type (str) --> which portion of data, e.g. train/test/valid (default 'train')
    - random_state (int) --> seed for repeating experiments (default 42)
    """
    file_name = f'data\\kaggle_ner\\{type}.txt'
    raw_df = pd.read_csv(file_name,sep=' ',header=None)
    return raw_df

if __name__ == '__main__':
    df = load_data()
    print(df.head())