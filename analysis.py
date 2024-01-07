import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import regexp_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("BooksDataSet.csv")
print(df.head(10))
df.drop(columns="Unnamed: 0", inplace=True)
print(df.head(10))

print(df["genre"].value_counts())

print(df.info())

sns.countplot(x="genre", data=df, palette="plasma", hue="genre")
plt.xticks(rotation=45)
plt.show()

print(df["summary"].iloc[1])

def cleantext(text):
    text = regexp_tokenize(text, pattern="\w+")
    text = " ".join(text)
    text = text.lower()
    return text

df["summary"] = df["summary"].apply(lambda x: cleantext(x))
print(df["summary"].iloc[1])
print(len(df["summary"].iloc[1]))


def showmostfrequentwords(text, no_of_words):
    allwords = " ".join([char for char in text])
    allwords = allwords.split()
    fdist = FreqDist(allwords)
    
    wordsdf = pd.DataFrame({"words":list(fdist.keys()), "count":list(fdist.values())})
    words = wordsdf.nlargest(columns="count", n=no_of_words)
    
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x="count", y="words", data=words, palette="cool", hue="count")
    ax.set(ylabel="words")
    plt.show()
    
    return wordsdf

wordsdf = showmostfrequentwords(df["summary"], 25)


eng_stopwords = stopwords.words("english")
def removestopwords(text):
    removedstopwords = [word for word in text.split() if word not in eng_stopwords]
    optext = " ".join(removedstopwords)
    return optext

df["summary"] = df["summary"].apply(lambda x: removestopwords(x))
print(df["summary"].iloc[1])
print(len(df["summary"].iloc[1]))


lematizer = WordNetLemmatizer()
def lematizing(sentence):
    stemsentence = ""
    for word in sentence.split():
        root_words = lematizer.lemmatize(word)
        stemsentence += root_words
        stemsentence += " "
    
    stemsentence = stemsentence.strip()
    return stemsentence

df["summary"] = df["summary"].apply(lambda x: lematizing(x))
print(df["summary"].iloc[1])
print(len(df["summary"].iloc[1]))


stemmer = PorterStemmer()
def stemming(sentence):
    stemmed_sentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemmed_sentence += stem
        stemmed_sentence += " "
        
    stemmed_sentence = stemmed_sentence.strip()
    return stemmed_sentence

df["summary"] = df["summary"].apply(lambda x: stemming(x))
print(df["summary"].iloc[1])
print(len(df["summary"].iloc[1]))

freq_df = showmostfrequentwords(df["summary"], 25)

encoder = LabelEncoder()
df["genre"] = encoder.fit_transform(df["genre"])
print(df["genre"].unique())

count_vec = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words="english")
bagofwordsvec = count_vec.fit_transform(df["summary"])
print(bagofwordsvec)

target = df["genre"]
print(target)

X_train, X_test, y_train, y_test = train_test_split(bagofwordsvec, target, test_size=0.2)

svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)
print(accuracy_score(y_test, svc_pred))

mb = MultinomialNB()
mb.fit(X_train, y_train)
mb_pred = mb.predict(X_test)
print(accuracy_score(y_test, mb_pred))

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(accuracy_score(y_test, rfc_pred))


X_train, X_test, y_train, y_test = train_test_split(df["summary"], df["genre"], test_size=0.2, random_state=557)

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.values.astype("U"))
X_test_tfidf = tfidf_vectorizer.transform(X_test.values.astype("U"))

svc_model = SVC()
svc_model.fit(X_train_tfidf, y_train)
svc_pred = svc_model.predict(X_test_tfidf)
print(accuracy_score(y_test, svc_pred))

mb = MultinomialNB()
mb.fit(X_train_tfidf, y_train)
mb_pred = mb.predict(X_test_tfidf)
print(accuracy_score(y_test, mb_pred))

rfc = RandomForestClassifier()
rfc.fit(X_train_tfidf, y_train)
rfc_pred = rfc.predict(X_test_tfidf)
print(accuracy_score(y_test, rfc_pred))


def testing(text, model):
    text = cleantext(text)
    text = removestopwords(text)
    text = lematizing(text)
    text = stemming(text)
    
    text_vector = tfidf_vectorizer.transform([text])
    predicted = model.predict(text_vector)
    return predicted

op = df["summary"].apply(lambda text: testing(text, mb))
print(op)

mapper = {
    0 : "Crime Fiction",
    1 : "Fantasy",
    2 : "History novel",
    3 : "Horror",
    4 : "Science Fiction",
    5 : "Thriller"
}

predicted_genres = []
for i in range(len(op)):
    idx_val = op[i][0]
    predicted_genres.append(mapper[idx_val])

comp_df = df.copy()
comp_df = comp_df[["summary", "genre"]]
print(comp_df)

comp_df["genre"] = comp_df["genre"].map(mapper)
comp_df["Predicted Genre"] = predicted_genres
print(comp_df)

sns.countplot(x="Predicted Genre", data=comp_df, palette="plasma", hue="Predicted Genre")
plt.xticks(rotation=45)
plt.show()

with open("bookgenremodel.pkl", "wb")as file:
    pickle.dump(mb, file)
    file.close()

with open("textvectorizer.pkl", "wb")as file2:
    pickle.dump(tfidf_vectorizer, file2)
    file2.close()
    