"""
***************
loading_data.py
***************
The purpose of this Python module is to illustrate how to load 
datasets from two modules, the first from Amazon Reviews and 
the second from Kaggle NER. 
"""
import kaggle_ner
import amazon_reviews

if __name__ == '__main__':

    ner_df = kaggle_ner.load_data()
    print(ner_df.head())

    path = r'data\kaggle_ner\train.txt'
    train_samples = kaggle_ner.load_sentences(path)

    path = r'data\kaggle_ner\test.txt'
    test_samples = kaggle_ner.load_sentences(path)
    
    path = r'data\kaggle_ner\valid.txt'
    valid_samples = kaggle_ner.load_sentences(path)
    
    samples = train_samples + test_samples

    schema = ['_'] + sorted({tag for sentence in samples 
                                for _, tag in sentence})

    from transformers import AutoConfig, TFAutoModelForTokenClassification

    MODEL_NAME = 'bert-base-cased' 

    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(schema))
    model = TFAutoModelForTokenClassification.from_pretrained(MODEL_NAME, 
                                                            config=config)
    model.summary()

    reviews_df = amazon_reviews.load_data()
    print(reviews_df.head())