"""
Load data of a category (e.g. appliances) downloaded from 
https://www.kaggle.com/alaakhaled/conll003-englishversion,
and produce a shuffled dataset required for BERT classification. 
Finally, demonstrate that BERT can be fine-tuned using the dataset of Amazon Reviews.
The code is based largely upon 
https://github.com/naveenjafer/BERT_Amazon_Reviews/blob/master/main.py
"""
import os
import numpy as np
import pandas as pd
import kaggle_ner
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoConfig, TFAutoModelForTokenClassification

class DataManager:

    def __init__(self):
        self.MODEL_NAME = 'bert-base-cased' 
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

    def tokenize_sample(self,sample):
        seq = [(subtoken, tag)
                for token, tag in sample
                for subtoken in self.tokenizer(token)['input_ids'][1:-1]]
        return [(3, 'O')] + seq + [(4, 'O')]
        
    def generate_arrays(self,samples):
        tag_index = {tag: i for i, tag in enumerate(self.schema)}
        tokenized_samples = list(tqdm(map(self.tokenize_sample, samples)))
        max_len = max(map(len, tokenized_samples))
        X = np.zeros((len(samples), max_len), dtype=np.int32)
        y = np.zeros((len(samples), max_len), dtype=np.int32)
        for i, sentence in enumerate(tokenized_samples):
            for j, (subtoken_id, tag) in enumerate(sentence):
                X[i, j] = subtoken_id
                y[i,j] = tag_index[tag]
        return X, y

    def process_sentences(self,filepath):
        """
        As in the Amazon reviews dataset, the Kaggle NER dataset
        involves multi-class classification of sentences. The major 
        difference, however, is how the content is labeled. For the 
        Amazon reviews, an entire was assigned a single label (the 
        overall score). Whereas here each word (or more accurately, 
        each token) in the sentences is assigned a tag (the entity 
        type, e.g. person or location).
        """
        final = [] # 
        sentences = [] # list of sentences belonging to the current document
        with open(filepath, 'r') as f:
            for line in f.readlines()[:100]: # MUST CHANGE THIS LIST TRUNCATION
            # for line in f.readlines():
                if (line == ('-DOCSTART- -X- -X- O\n') or line == '\n'):
                    if len(sentences) > 0:
                        final.append(sentences)
                        sentences = []
                else:
                    l = line.split(' ')
                    sentences.append((l[0], l[3].strip('\n')))
        random.seed(10)
        random.shuffle(final)
        return final

    def construct_arrays(self):
        """
        The shape of the arrays (or tensors) here will differ from 
        those resulting from the Amazon review dataset.
        Each X array is comprised of the tokens for each word/punctuation in the sentence, plus padding.
        Each y array is comprised of the numeric keys for the NER tag of each word, plus padding.
        """
        self.X_train, self.y_train = self.generate_arrays(self.train_samples)
        self.X_test, self.y_test = self.generate_arrays(self.test_samples)
        self.X_valid, self.y_valid = self.generate_arrays(self.valid_samples)

    def read_files(self):
        """Store lists of sentences from each dataset's text file."""
        base_path = os.path.join('data','kaggle_ner')
        self.train_samples = self.process_sentences(os.path.join(base_path,'train.txt'))
        self.test_samples = self.process_sentences(os.path.join(base_path,'test.txt'))
        self.valid_samples = self.process_sentences(os.path.join(base_path,'valid.txt'))

    def generate_schema(self):
        """Gather all possible tags."""        
        samples = self.train_samples + self.test_samples
        self.schema = ['_'] + sorted({tag for sentence in samples 
                                    for _, tag in sentence})

    def load_data(self):
        """
        Based on the data from the examples of NER in the text files, 
        produce the arrays needed for BERT classification.
        """
        self.read_files()
        self.generate_schema()
        self.construct_arrays()

if __name__ == '__main__':

    # Load data
    dm = DataManager()
    dm.load_data() # store data as attributes (e.g. dm.X_train)

    # Init BERT model
    MODEL_NAME = dm.MODEL_NAME
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(dm.schema))
    model = TFAutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, config=config)

    # Train model
    EPOCHS=5
    BATCH_SIZE=8
    optimizer = tf.keras.optimizers.Adam(lr=0.000001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    history = model.fit(tf.constant(dm.X_train), tf.constant(dm.y_train),
                        validation_data=(dm.X_test, dm.y_test), 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE)

    # Plot loss
    plt.figure(figsize=(14,8))
    plt.title('Losses')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Valid Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(os.path.join('images',"loss.png"))

    # Plot accuracy
    plt.figure(figsize=(14,8))
    plt.title('Accuracies')
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Valid Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(os.path.join('images',"accuracy.png"))

    # Print results
    [loss, accuracy] = model.evaluate(dm.X_valid, dm.y_valid)
    print("Loss:%1.3f, Accuracy:%1.3f" % (loss, accuracy))