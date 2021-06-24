"""
classify_NER.py

CREDITS:
This code is based upon 
https://www.kaggle.com/alincijov/conll-huggingface-named-entity-recognition

GOAL:
The purpose of this Python module is to demonstrate 
that BERT can be fine-tuned using the dataset of Kaggle NER.

SPECIFICATIONS:
In this script, it will be made clear 
(1) how this dataset differs from that of Amazon reviews, and 
(2) how the resulting tensor differs from that of Amazon reviews.
"""
import numpy as np
import pandas as pd
import kaggle_ner
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoConfig, TFAutoModelForTokenClassification

if __name__ == '__main__':

    train_samples,test_samples,valid_samples,samples,schema = kaggle_ner.load_data()

    # *************** TOKENIZE ***************
    MODEL_NAME = 'bert-base-cased' 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_sample(sample):
        seq = [
                (subtoken, tag)
                for token, tag in sample
                for subtoken in tokenizer(token)['input_ids'][1:-1]
            ]
        return [(3, 'O')] + seq + [(4, 'O')]

    def preprocess(samples):
        tag_index = {tag: i for i, tag in enumerate(schema)}
        tokenized_samples = list(tqdm(map(tokenize_sample, samples)))
        max_len = max(map(len, tokenized_samples))
        X = np.zeros((len(samples), max_len), dtype=np.int32)
        y = np.zeros((len(samples), max_len), dtype=np.int32)
        for i, sentence in enumerate(tokenized_samples):
            for j, (subtoken_id, tag) in enumerate(sentence):
                X[i, j] = subtoken_id
                y[i,j] = tag_index[tag]
        return X, y

    X_train, y_train = preprocess(train_samples)
    X_test, y_test = preprocess(test_samples)
    X_valid, y_valid = preprocess(valid_samples)

    # *************** MODEL ***************
    MODEL_NAME = 'bert-base-cased' 
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(schema))
    model = TFAutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, config=config)
    model.summary()


    
    # *************** TRAINING ***************
    EPOCHS=5
    BATCH_SIZE=8

    optimizer = tf.keras.optimizers.Adam(lr=0.000001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    history = model.fit(tf.constant(X_train), tf.constant(y_train),
                        validation_data=(X_test, y_test), 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE)

    # *************** RESULTS ***************
    plt.figure(figsize=(14,8))
    plt.title('Losses')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Valid Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show("training_results.png")


    # *************** Validation ***************
    [loss, accuracy] = model.evaluate(X_valid, y_valid)
    print("Loss:%1.3f, Accuracy:%1.3f" % (loss, accuracy))