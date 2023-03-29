# Machine Learning Project: Tweet Classification

Introduction

The goal of this project was to build a fine-tuned Machine Learning model for binary classification of tweets based on their sentiment. The model is designed to recognize if a tweet would have contained a ":)" smiley or a ":(" smiley, which is a proxy for classifying whether the tweet is positive or negative. Five dataset were provided: two full training datasets containing roughly 1 250 000 tweets each, with respectively positive and negative tweets, two samples of the training tweets containing about 10\% of the full data, and a test dataset of 10 000 unlabeled tweets to submit our predictions with label encoding as +1 and -1 for ':)' (positive) and ':(' (negative) respectively. To evaluate our models we used randomized splitting on the partial and full training data as well as cross-validation, then final testing with submission on AICrowd (www.aicrowd.com) which would then compare the obtained labels with the expected ones and rank based on the accuracy and F1-Score.

### Folder structure

.
├── embeddings.py              # Methods to embed text to vectors using different embeddings such as GloVe and Word2Vec

├── models.py                  # Methods to train, test and predict various ML models and Recurrent Neural Networks.

├── fasttext_model.py          # Methods to label tweets according to fasttext syntax and run the fasttext model

├── preprocessing.py           # Methods for pre-processing

└── run.ipynb                  # Main file with most combinations of embeddings/models that we tried

### Table of content

Python files containing embeddings and models in order to classify a dataset of tweets based on sentiment.
A PDF report explaining our work process and discoveries, as well as the accuracies and F1-Scores we obtained.

Embeddings: GloVe, Word2Vec, Fasttext, BERT
Models: Logistic Regression, Linear SVM, MLP Classifier, Fasttext
Recurrent Neural Networks: LSTM, GRU

Branch "jacopo" contains exploration files on TF-IDF and n-grams.
In the report, we had to limit the number of configurations we presented, we did not mention Decision Tree Classifier, Ensemble Methods nor Boosting methods but all experiments are detailed in the secondary branch 'jacopo', divided in versions. Note how our best submission is at the end of exploration_v2_Jacopo file. By going through the printed results and comments you can see details on time, testing, hyperparameters tuning. 

### How to run ?

Running the file run.ipynb will sequentially run every combinations of embeddings and models we have tested, giving corresponding accuracies and F1-scores, as well as writing submission files. Some combinations might be particularly expensive and time-consuming to run, such as the Neural Networks and BERT that we implemented but
suffers from OOM errors. The file is hence not really meant to be run at once, but rather having each of its sections running individually.






