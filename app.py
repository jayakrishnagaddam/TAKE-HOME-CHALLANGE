from flask import Flask, request, jsonify, render_template
from flask_pymongo import PyMongo
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

app.config['MONGO_URI'] = 'mongodb+srv://2100090162:manigaddam@deepsheild.kzgpo9p.mongodb.net/EMA'
mongo = PyMongo(app)

nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    text = ' '.join(token.text.lower() for token in nlp(text) if not token.is_punct and not token.like_num)

    text = ' '.join(word for word in text.split() if word not in STOP_WORDS)
    return text

def retrieve_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

class QuestionAnsweringModel:
    def __init__(self, data):
        self.data = data
        self.vectorizer = TfidfVectorizer()

        self.sentences = [preprocess_text(sent) for sent in retrieve_sentences(data)]
        self.vectorizer.fit(self.sentences)
        self.tfidf_matrix = self.vectorizer.transform(self.sentences)

    def answer_question(self, question):
        question = preprocess_text(question)
        question_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, self.tfidf_matrix)
        most_similar_idx = similarities.argmax()
        return self.sentences[most_similar_idx]

model = None
initialized = False

def initialize_model():
    global model, initialized
    collection = mongo.db.Data
    data_document = collection.find_one({})
    if data_document:
        content = data_document.get('Content')
        model = QuestionAnsweringModel(content)
        initialized = True
    else:
        raise ValueError("No data found in MongoDB.")

def ensure_model_initialized():
    global initialized
    if not initialized:
        initialize_model()

@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    ensure_model_initialized()
    collection = mongo.db.Data
    data_from_db = collection.find_one({})
    return jsonify(data_from_db)

@app.route('/query_model', methods=['POST'])
def query_model():
    ensure_model_initialized()
    global model
    if not model:
        return jsonify({'response': 'Model not initialized.'}), 500

    query = request.json.get('query')
    answer = model.answer_question(query)
    return jsonify({'response': answer})

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
