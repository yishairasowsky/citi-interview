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
    #     self.MODEL_NAME = 'bert-base-cased' 
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

    # def tokenize_sample(self,sample):
    #     seq = [(subtoken, tag)
    #             for token, tag in sample
    #             for subtoken in self.tokenizer(token)['input_ids'][1:-1]]
    #     return [(3, 'O')] + seq + [(4, 'O')]
        
    # def generate_arrays(self,samples):
    #     # assign numbers to NER tag
    #     tag_index = {tag: i for i, tag in enumerate(self.schema)} 
    #     tokenized_samples = list(tqdm(map(self.tokenize_sample, samples)))
    #     # implement padding
    #     max_len = max(map(len, tokenized_samples))
    #     X = np.zeros((len(samples), max_len), dtype=np.int32)
    #     y = np.zeros((len(samples), max_len), dtype=np.int32)
    #     # fill in 
    #     for sentence_idx, tokenized_sentence in enumerate(tokenized_samples):
    #         for position_in_sentence, (subtoken_id, tag) in enumerate(tokenized_sentence):
    #             X[sentence_idx, position_in_sentence] = subtoken_id
    #             y[sentence_idx,position_in_sentence] = tag_index[tag]
    #     return X, y

    # def process_sentences(self,filepath):
    #     """
    #     As in the Amazon reviews dataset, the Kaggle NER dataset
    #     involves multi-class classification of sentences. The major 
    #     difference, however, is how the content is labeled. For the 
    #     Amazon reviews, an entire was assigned a single label (the 
    #     overall score). Whereas here each word (or more accurately, 
    #     each token) in the sentences is assigned a tag (the entity 
    #     type, e.g. person or location).
    #     """
    #     tagged_sentences = [] # 
    #     doc_sentences = [] # list of sentences belonging to the current document
    #     with open(filepath, 'r') as f:
    #         for line in f.readlines():
    #             # upon reaching a new group of sentences
    #             if (line == ('-DOCSTART- -X- -X- O\n') or line == '\n'):
    #                 if len(doc_sentences) > 0:
    #                     # store content from previous group
    #                     tagged_sentences.append(doc_sentences)
    #                     # reinitialize to gather next group of sentences
    #                     doc_sentences = []
    #             else:
    #                 # store important information, i.e. text and NER tag 
    #                 entity_attributes = line.split(' ')
    #                 entity_text = entity_attributes[0]
    #                 entity_tag = entity_attributes[3]
    #                 doc_sentences.append((entity_text, entity_tag.strip('\n')))
    #     random.seed(10)
    #     random.shuffle(tagged_sentences)
    #     return tagged_sentences

    # def construct_arrays(self):
    #     """
    #     Here the shape of the arrays (or tensors) will differ from 
    #     those resulting from the Amazon review dataset.
        
    #     Each X array is comprised of the tokens for each 
    #     word/punctuation in the sentence.
        
    #     Each y array is comprised of the numeric keys for the NER 
    #     tag of each word.
        
    #     Any array has padding on the right side to make the length 
    #     of all sentences uniform.
    #     """
    #     self.X_train, self.y_train = self.generate_arrays(self.train_samples)
    #     self.X_test, self.y_test = self.generate_arrays(self.test_samples)
    #     self.X_valid, self.y_valid = self.generate_arrays(self.valid_samples)

    # def read_files(self):
    #     """Store lists of sentences from each dataset's text file."""
    #     base_path = os.path.join('data','kaggle_ner')
    #     self.train_samples = self.process_sentences(os.path.join(base_path,'train.txt'))
    #     self.test_samples = self.process_sentences(os.path.join(base_path,'test.txt'))
    #     self.valid_samples = self.process_sentences(os.path.join(base_path,'valid.txt'))

    # def generate_schema(self):
    #     """Gather all possible tags."""        
    #     samples = self.train_samples + self.test_samples
    #     self.schema = ['_'] + sorted({tag for sentence in samples 
    #                                 for _, tag in sentence})

    # def load_data(self):
    #     """
    #     Based on the data from the examples of NER in the text files, 
    #     produce the arrays needed for BERT classification.
    #     """
    #     self.read_files()
    #     self.generate_schema()
    #     self.construct_arrays()

    # def load_data(selfproduct_category='Appliances', row_limit=None, random_state=42):
    
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
