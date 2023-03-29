from fasttext import train_supervised 

def fasttext_label(df):
    df_copy = df.copy()
    df_copy["label"] = df_copy["label"].apply(lambda label: "__label__1" if label == 1 else "__label__0")
    df_copy["tweets"] = df_copy["tokens"].apply(lambda x: " ".join(t for t in x))
    return df_copy  


def train_test_fasttext_model(df_train,df_test):
    train_file = 'data/fasstext_train.txt'
    df_train.to_csv(train_file, header=None, index=False, sep=' ', columns=["label","tweets"])
    test_file =  'data/fasstext_test.txt'
    df_test.to_csv(test_file, header=None, index=False, sep=' ', columns=["tweets"])    

    model = train_supervised(input=train_file)

    predictions = []
    file1 = open('data/fasstext_test.txt', 'r', encoding="utf-8")
    lines = file1.readlines()
    for line in lines:
        pred = model.predict(line.strip())[0][0]
        if pred == "__label__0":
            predictions.append(-1)
        else:    
            predictions.append(1)
    return predictions   