import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}


def load_datasets(full = False):

    """
    neg_path = 'drive/My Drive/Colab Notebooks/twitter-datasets/train_neg.txt'
    pos_path = 'drive/My Drive/Colab Notebooks/twitter-datasets/train_pos.txt'
    """
    neg_path = "data/twitter-datasets/train_neg.txt"
    pos_path = "data/twitter-datasets/train_pos.txt"
    if full:
        neg_path = "data/twitter-datasets/train_neg_full.txt"
        pos_path = "data/twitter-datasets/train_pos_full.txt"
    df_train_neg = pd.read_csv(neg_path, delimiter="\t", header=None, names = ['tweets'], on_bad_lines="skip")
    df_train_pos = pd.read_csv(pos_path, delimiter="\t", header=None, names = ['tweets'], on_bad_lines="skip")
    df_train_neg["label"] = -1
    df_train_pos["label"] = 1

    df_train = pd.concat([df_train_pos,df_train_neg])
    
    df_test = pd.read_csv("data/twitter-datasets/test_data.txt", delimiter="\t", header=None, names = ['tweets'], on_bad_lines="skip")
    """
    df_test = pd.read_csv("data/twitter-datasets/test_data.txt", delimiter="\t", header=None, names = ['tweets'], on_bad_lines="skip")
    """
    df_test["tweets"] = df_test["tweets"].apply(lambda row: row.split(",",2)[1])

    return df_train, df_test

def remove_tags(df):
    df_cleaned = df.copy()
    df_cleaned['tweets'] = df_cleaned['tweets'].apply(lambda tweet: tweet.replace("<url>","").replace("<user>",""))
    return df_cleaned

def unslang(df):
    df_unslang = df.copy()
    df_unslang['tweets'] = df_unslang['tweets'].apply(lambda tweet: ' '.join([abbreviations.get(word,word) for word in tweet.split()]))
    return df_unslang

def smileys_to_word(df):
    df = df.replace(r"[8:=;]['`\-]?[)dD]+|[(dD]+['`\-]?[8:=;]", "happy" , regex=True)
    df = df.replace(r"[8:=;]['`\-]?[(]+|[)]+['`\-]?[8:=;]", "sad" , regex=True)
    df = df.replace(r"[8:=;]['`\-]?[\/|l]+", "neutral" , regex=True)
    df = df.replace(r"<3" , "love" , regex=True)
    return df
    
def unelongate(df):
    #shorten words with multiple chars
    df = df.replace(r"\b(\S*?)(.)\2{2,}\b", r"\1", regex=True)
    #shorten repeated punct
    df = df.replace(r"([!?.]){2,}" , r"\1" , regex=True)
    return df

def uncase(df):
    df = df.apply(lambda x: x.astype(str).str.lower())
    return df

def tokenize_and_preprocess(df, stop_words = False, stemming = False, lemmatization = False, unslang_bool = False, remove_tags_bool = False, unelongate_bool = False, uncase_bool = False, smiley_to_word_bool = False):
    
    df_cleaned = df.copy()
    
    if remove_tags_bool:
        df_cleaned = remove_tags(df_cleaned)        
    if uncase_bool:
        df_cleaned = uncase(df_cleaned)
    if unelongate_bool:
        df_cleaned = unelongate(df_cleaned)
    if smiley_to_word_bool:
        df_cleaned = smileys_to_word(df_cleaned)
    if unslang_bool:
        df_cleaned = unslang(df_cleaned)
              
    df_cleaned['tokens'] = df_cleaned['tweets'].apply(lambda tweet: word_tokenize(tweet)) 
    # remove stop words
    if stop_words:
        stop_words = stopwords.words('english')
        df_cleaned['tokens'] = df_cleaned['tokens'].apply(lambda tokens: [token for token in tokens if token.lower() not in stop_words])
    # stemming
    if stemming:
        ps = PorterStemmer()
        df_cleaned['tokens'] = df_cleaned['tokens'].apply(lambda tokens: [ps.stem(token) for token in tokens])
    # lemmatization
    if lemmatization:
        wordnet_lemmatizer = WordNetLemmatizer()
        df_cleaned['tokens'] = df_cleaned['tokens'].apply(lambda tokens: [wordnet_lemmatizer.lemmatize(token) for token in tokens])
    # remove the tweets columns
    return df_cleaned  

