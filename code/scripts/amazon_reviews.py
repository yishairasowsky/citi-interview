"""
The purpose of this Python module is 
to take data downloaded from one of the categories at
http://deepyeti.ucsd.edu/jianmo/amazon/index.html,
and eventually output a shuffled dataset 
required for BERT classification. 
CREDITS:
This code is based upon 
https://github.com/naveenjafer/BERT_Amazon_Reviews/blob/master/main.py

GOAL:
The purpose of this Python module is to demonstrate 
that BERT can be fine-tuned using the dataset of Amazon Reviews.

SPECIFICATIONS:
In this script, it will be made clear 
(1) how this dataset differs from that of Kaggle NER, and 
(2) how the resulting tensor differs from that of Kaggle NER.

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
    
    import torch
    import amazon_reviews
    from transformers import BertTokenizer
    from torch.utils.data import TensorDataset
    from transformers import BertForSequenceClassification
    
    df = load_data()
    print(df.head())


    df = amazon_reviews.load_data(row_limit=500)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)

    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type=='train'].text.values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type=='val'].text.values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type=='train'].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type=='val'].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
