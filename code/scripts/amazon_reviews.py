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
import torch
import numpy as np
import random
import pandas as pd

from tqdm import tqdm
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
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
        self.label_dict = {possible_label:possible_label - 1 for possible_label in possible_labels}
        self.df['label'] = self.df.overall.replace(self.label_dict)
    
    def raw_data(self,product_category):
        # all raw data
        file_name = os.path.join(r'data/amazon_reviews',f'{product_category}.json.gz')
        self.df = pd.read_json(file_name,compression='infer',lines=True) 
    
    def select_input(self):
        # choose relevant features
        self.df["text"] = self.df["summary"] + ' ' + self.df["reviewText"]
        self.df["text"] = self.df["text"].str.lower()

    def train_val_split(self):
        X_train, X_val, _, _ = train_test_split(self.df.index.values, 
                                                        self.df.label.values, 
                                                        test_size=0.15, 
                                                        random_state=42, 
                                                        stratify=self.df.label.values)
        # label data type in column
        self.df['data_type'] = ['not_set']*self.df.shape[0] 
        self.df.loc[X_train, 'data_type'] = 'train'
        self.df.loc[X_val, 'data_type'] = 'val'

    def tokenize(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                            do_lower_case=True)

        self.encoded_data_train = tokenizer.batch_encode_plus(
            self.df[self.df.data_type=='train'].text.values, 
            add_special_tokens=True, 
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=256, 
            return_tensors='pt'
        )

        self.encoded_data_val = tokenizer.batch_encode_plus(
            self.df[self.df.data_type=='val'].text.values, 
            add_special_tokens=True, 
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=256, 
            return_tensors='pt'
        )

    def generate_tensors(self):
        """
        Returns two dataset tensors, each of which is comprised of three tensors.
        
        In each tensor, the first index indicates the relevant 
        sentence, while the second indicates the location of the 
        relevant token in that sentence.
        
        In the first tensor, each value represents the relevant 
        token's numeric ID.
        
        In the second tensor, each value represents whether that 
        token has nay meaningful content (1) or not (0). The 
        purpose is so that only nontrivial tokens should contribute 
        to our training.
        
        In the third tensor, each value represents the which of the 
        multiple classes is the correct overall score of the review.

        All tensors are padded to the right, to ensure uniform 
        length of all sentences.
        """
        
        input_ids_train = self.encoded_data_train['input_ids']
        attention_masks_train = self.encoded_data_train['attention_mask']
        labels_train = torch.tensor(self.df[self.df.data_type=='train'].label.values)

        input_ids_val = self.encoded_data_val['input_ids']
        attention_masks_val = self.encoded_data_val['attention_mask']
        labels_val = torch.tensor(self.df[self.df.data_type=='val'].label.values)

        self.dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
        self.dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

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
        self.raw_data(product_category)
        self.select_input()
        self.set_target()

        # As in the Kaggle NER dataset, this Amazon reviews dataset
        # involves multi-class classification of sentences. The major 
        # difference, however, is how the content is labeled. For the 
        # Kaggle NER sentences, each word (or more accurately, 
        # each token) in the sentences is assigned a tag (the entity 
        # type, e.g. person or location). Whereas here an entire was
        # assigned a single label (the overall score).

        # remove unwanted content
        self.select_columns()
        self.drop_rows()
        
        # prepare for classification
        self.shuffle_rows()
        self.limit_rows(row_limit)
        self.train_val_split()

        self.tokenize()
        self.generate_tensors()
        

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals
    
if __name__ == '__main__':
    
    dm = DataManager()
    dm.load_data(row_limit=300)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=len(dm.label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False)

    batch_size = 3

    dataloader_train = DataLoader(dm.dataset_train, 
                                sampler=RandomSampler(dm.dataset_train), 
                                batch_size=batch_size)

    dataloader_validation = DataLoader(dm.dataset_val, 
                                    sampler=SequentialSampler(dm.dataset_val), 
                                    batch_size=batch_size)


    optimizer = AdamW(model.parameters(),
                    lr=1e-5, 
                    eps=1e-8)

    epochs = 5

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train)*epochs)

    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(device)
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    # torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
    torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')
    _, predictions, true_vals = evaluate(dataloader_validation)
    results = accuracy_per_class(predictions, true_vals)
    print(results)
