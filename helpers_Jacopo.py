# ====================================================================#
# Helper Functions
# ====================================================================#

import pandas as pd
import numpy as np
import pickle

def load_test_data(path='data/twitter-datasets/test_data.txt'):
    # Load the test data
    df_test = pd.read_csv(path, sep='\t', names=['line'])
    df_test['tweet'] = df_test['line'].apply(lambda x: x.split(',', 1)[1])
    df_test['id'] = df_test['line'].apply(lambda x: x.split(',', 1)[0])
    df_test = df_test.drop('line', axis=1)
    print('Test set: ', df_test.shape)
    return df_test


def load_train_data(path_pos='data/twitter-datasets/train_pos_full.txt', path_neg='data/twitter-datasets/train_neg_full.txt'):
    # Load data, txt as csv
    #data_path = 'data/twitter-datasets/'
    df_train_pos = pd.read_csv(path_pos, sep = '\t', names = ['tweet'])
    df_train_pos['label'] = 1
    df_train_neg = pd.read_csv(path_neg, sep = '\t', names = ['tweet'])
    df_train_neg['label'] = 0
    df_train = pd.concat([df_train_pos, df_train_neg])
    print('Train set: ', df_train.shape)
    print('Train set positives: ', df_train_pos.shape)
    print('Train set negatives: ', df_train_neg.shape)
    return df_train


def build_feature_matrix(df, vocab, embeddings, mode='avg'):
    X = np.zeros((df.shape[0], embeddings.shape[1]))
    for i, tweet in enumerate(df['tweet']):
        words = tweet.split()
        for word in words:
            if word in vocab:
                X[i] += embeddings[vocab[word]]
        if mode == 'avg':
            X[i] /= len(words)
        elif mode == 'sum':
            pass
        else:
            raise ValueError('Unknown mode: {}'.format(mode))
    return X

def predict_test_data(X_test, classifier, filename='submission.csv'):
    # Predict test data and save to csv
    y_pred = classifier.predict(X_test)
    df_test = pd.DataFrame()
    df_test['Prediction'] = y_pred
    df_test.rename(columns={'id': 'Id'}, inplace=True)
    df_test['Prediction'] = df_test['Prediction'].apply(lambda x: -1 if x == 0 else x)
    df_test.to_csv(filename, columns=['Id', 'Prediction'], index=False)
    return None

    