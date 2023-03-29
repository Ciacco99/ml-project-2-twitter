# Logistic reg imports
from sklearn.linear_model import LogisticRegression

# SVM imports
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# K-NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

# NN imports
from sklearn.neural_network import MLPClassifier

"""
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,Dropout,Embedding,CuDNNLSTM,Bidirectional,Conv1D,MaxPooling1D,GRU,Flatten
"""


def train_test_model(model_to_use,x_train,y_train,x_test):
    
    if model_to_use == "LogisticRegression":
        logisticRegr = LogisticRegression(max_iter=4000)
        logisticRegr.fit(x_train, y_train) 
        return logisticRegr.predict(x_test)
    
    if model_to_use == "SVM": #Grid Search ??
        # Use linear SVM for optimization (text classification is often linear)
        lin_clf = svm.LinearSVC(max_iter=1000)
        lin_clf.fit(x_train,y_train)
        return lin_clf.predict(x_test)
    
    if model_to_use == "MLPClassifier":

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(30, 12), random_state=1,max_iter=4000)
        
        preds = clf.predict(x_test)
        loss = clf.loss_curve_
        return preds, loss, best_params
    
def train_test_rnn(model_to_use, embedding_matrix, vocab_size, pad_seq, y, test_seq): 
    
    if model_to_use == "RNN LSTM":   
        
        model=Sequential()
        model.add(Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=200, trainable=False))
        model.add(LSTM(100,activation='relu',return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(100,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(4,activation='softmax'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(pad_seq, y, epochs=20, batch_size=1024, verbose=1)
        
       
    if model_to_use == "RNN GRU":  
        
        model = Sequential()
        embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=200 , trainable=False)
        model.add(embedding_layer)
        model.add(GRU(128, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation='sigmoid'))
        
    if  mode_to_use == "LSTM+GRU":
        
        model = Sequential()
        model.add(Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=200, trainable=False))
        model.add(LSTM(100,return_sequences=True,activation="sigmoid"))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation = "sigmoid"))
        model.add(GRU(128,return_sequences = True))
        model.add(Dense(100, activation = "sigmoid"))
        model.add(Dense(1, activation = "sigmoid"))
        
    return model.predict(test_seq, batch_size=1024, verbose=1)
    
            
        