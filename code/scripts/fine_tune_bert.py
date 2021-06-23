"""
*****************
fine_tune_bert.py
*****************
The purpose of this Python module is to demonstrate 
that BERT can be fine-tuned using the two datasets, 
the first from Amazon Reviews and the second from Kaggle NER. 
In this script, it will be made clear 
(1) how these two datasets are different, and 
(2) what the differences in the resulting tensors are.
"""
import kaggle_ner
import amazon_reviews

if __name__ == '__main__':

    ner_df = kaggle_ner.load_data()
    print(ner_df.head())

    reviews_df = amazon_reviews.load_data()
    print(reviews_df.head())