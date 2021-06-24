"""
Provide function to load and shuffle a category of data
(e.g. appliances) downloaded from 
http://deepyeti.ucsd.edu/jianmo/amazon/index.html,
and eventually output a shuffled dataset 
required for BERT classification. 

The code is based upon 
https://github.com/naveenjafer/BERT_Amazon_Reviews/blob/master/main.py
"""
import os
import pandas as pd

from sklearn.model_selection import train_test_split

class DataManager:

    def __init__(self):
        pass

    def limit_rows(self,row_limit):
        if row_limit:
            self.df = self.df.iloc[:row_limit,:]

    def shuffle_rows(self):
        # shuffle rows
        self.df = self.df.sample(frac=1,random_state=42) 
    
    def drop_rows(self):
        # remove unwanted rows
        self.df = self.df.dropna(subset=['label', 'text'])
    
    def select_columns(self):
        # select wanted columns
        cols = ['text','label']
        self.df = self.df[cols]
    
    def set_target(self):
        # organize target variable
        possible_labels = self.df.overall.unique()
        label_dict = {possible_label:possible_label - 1 for possible_label in possible_labels}
        self.df['label'] = self.df.overall.replace(label_dict)
    
    def read_raw(self,product_category):
        # all raw data
        file_name = os.path.join(r'data/amazon_reviews',f'{product_category}.json.gz')
        self.df = pd.read_json(file_name,compression='infer',lines=True) 
    
    def select_input(self):
        # select relevant input features
        self.df["text"] = self.df["summary"] + ' ' + self.df["reviewText"]
        self.df["text"] = self.df["text"].str.lower()
        
    def load_data(self,row_limit=None,product_category='Appliances'):
        """
        Get shuffled datasets ready for BERT classification. 

        Parameters:
            product_category (str): Type of item purchased.
            random_state (int): Seed for repeating experiments.
            row_limit (int): Maximum number of rows allowed.
            
        Returns:
            df (pd.DataFrame): A table containing training and validation samples.
        """
        # choose data
        self.read_raw(product_category)
        self.select_input()
        self.set_target()

        # remove unwanted content
        self.select_columns()
        self.drop_rows()
        
        # prepare for classification
        self.shuffle_rows()
        self.limit_rows(row_limit)

        # split samples into training versus validation
        X_train, X_val, _, _ = train_test_split(self.df.index.values, 
                                                        self.df.label.values, 
                                                        test_size=0.15, 
                                                        random_state=42, 
                                                        stratify=self.df.label.values)
        # label data type in column
        self.df['data_type'] = ['not_set']*self.df.shape[0] 
        self.df.loc[X_train, 'data_type'] = 'train'
        self.df.loc[X_val, 'data_type'] = 'val'

        return self.df

if __name__ == '__main__':
    
    import torch
    from transformers import BertTokenizer
    from torch.utils.data import TensorDataset

    dm = DataManager()
    df = dm.load_data(row_limit=500)

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
