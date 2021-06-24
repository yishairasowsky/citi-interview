"""
*************
kaggle_ner.py
*************
The purpose of this Python module is 
to take data downloaded from one of the categories at
https://www.kaggle.com/alaakhaled/conll003-englishversion,
and eventually output a shuffled dataset required for BERT classification. 
"""
import os
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


def process_sentences(filepath):
    final = []
    sentences = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if (line == ('-DOCSTART- -X- -X- O\n') or line == '\n'):
                if len(sentences) > 0:
                    final.append(sentences)
                    sentences = []
            else:
                l = line.split(' ')
                sentences.append((l[0], l[3].strip('\n')))
    return final

def load_data():
    # base_path = '../input/conll003-englishversion/'
    base_path = os.path.join('data','kaggle_ner')

    train_samples = process_sentences(os.path.join(base_path,'train.txt'))
    test_samples = process_sentences(os.path.join(base_path,'test.txt'))
    valid_samples = process_sentences(os.path.join(base_path,'valid.txt'))

    samples = train_samples + test_samples

    schema = ['_'] + sorted({tag for sentence in samples 
                                for _, tag in sentence})

    return train_samples,test_samples,valid_samples,samples,schema

if __name__ == '__main__':
    train_samples,test_samples,valid_samples,samples,schema = load_data()
    print('done')
    # print(df.head())