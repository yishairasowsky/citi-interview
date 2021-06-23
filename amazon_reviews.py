"""
*****************
amazon_reviews.py
*****************
The purpose of this Python module is 
to take data downloaded from one of the categories at
http://deepyeti.ucsd.edu/jianmo/amazon/index.html,
and eventually output a shuffled dataset required for BERT classification. 
"""
import pandas as pd

def load_data(product_category='Appliances', random_state=42):
    """
    Returns a dataframe containing Amazon reviews.
    Args:
    - product_category (str) --> type of item purchased (default 'Appliances')
    - random_state (int) --> seed for repeating experiments (default 42)
    """
    file_name = f'data\\amazon_reviews\\{product_category}.json.gz'
    raw_df = pd.read_json(file_name,compression='infer',lines=True)
    shuffled_df = raw_df.sample(frac=1,random_state=random_state)
    return shuffled_df

if __name__ == '__main__':
    df = load_data()
    print(df.head())