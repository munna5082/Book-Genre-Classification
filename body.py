from flask import Flask, render_template, request
import pickle
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

with open("bookgenremodel.pkl", "rb")as file:
    model = pickle.load(file)

with open("textvectorizer.pkl", "rb")as file2:
    textvectorizer = pickle.load(file2)

def cleantext(text):
    text = regexp_tokenize(text, pattern="\w+")
    text = " ".join(text)
    text = text.lower()
    return text

def removestopwords(text):
    eng_stopwords = stopwords.words("english")
    removedstopwords = [word for word in text.split() if word not in eng_stopwords]
    optext = " ".join(removedstopwords)
    return optext

def lematizing(sentence):
    lematizer = WordNetLemmatizer()
    stemsentence = ""
    for word in sentence.split():
        root_words = lematizer.lemmatize(word)
        stemsentence += root_words
        stemsentence += " "
    
    stemsentence = stemsentence.strip()
    return stemsentence

def stemming(sentence):
    stemmer = PorterStemmer()
    stemmed_sentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemmed_sentence += stem
        stemmed_sentence += " "
        
    stemmed_sentence = stemmed_sentence.strip()
    return stemmed_sentence

def testing(text, model, textvectorizer):
    text = cleantext(text)
    text = removestopwords(text)
    text = lematizing(text)
    text = stemming(text)
    
    text_vector = textvectorizer.transform([text])
    predicted = model.predict(text_vector)

    mapper = {0 : "Crime Fiction", 1 : "Fantasy", 2 : "History novel", 3 : "Horror", 4 : "Science Fiction", 5 : "Thriller"}

    return mapper[predicted[0]]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        text = request.form["summary"]
        result = testing(text, model, textvectorizer)
        return render_template("output.html", text_data = str(text)[:100], output = result)



if __name__ == "__main__":
    app.run()