"""
classify_reviews.py

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
import amazon_reviews
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

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
