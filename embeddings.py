import numpy as np
import csv
from gensim.models import Word2Vec 
import multiprocessing


def load_glove_embeddings():
    glove_embeddings = dict()
    #with open("data/glove.6B.100d.txt", "r", encoding="utf-8") as f:
    with open("data/glove.twitter.27B.100d.txt", "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = coefs
    return glove_embeddings  


def word2vec(tokens,model):
    embeddings = []
    for token in tokens:
        if token in model.wv:
            embeddings.append(model.wv[token])
    if len(embeddings) == 0:
        return np.zeros(100)
    return np.average (embeddings,axis=0)

def glove2vec(tokens,glove_embeddings):
    embeddings = []
    for token in tokens:
        word = token.lower()
        embedding = glove_embeddings.get(token)
        if embedding is not None:
            embeddings.append(embedding) 
    if len(embeddings) == 0:
        return np.zeros(100)
    return np.average(embeddings,axis=0) 

def tweet2vec(df, embedding_type):
    if embedding_type == "glove":
        glove_embeddings = load_glove_embeddings()
        df["vectors"] = df["tokens"].apply(lambda tokens: glove2vec(tokens,glove_embeddings))
        return df
    if embedding_type == "word2vec":
        cores = multiprocessing.cpu_count() 
        tokens = df["tokens"].tolist()
        model = Word2Vec(tokens, min_count = 5, window = 10, vector_size = 100, workers=cores-1)  
        df["vectors"] = df["tokens"].apply(lambda tokens: word2vec(tokens,model))
        return df

def write_submission(predictions,filename):
    path = "data/" + filename
    ids=[i for i in range(1,len(predictions)+1)]
    with open(path, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, predictions):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})         
