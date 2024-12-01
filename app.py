from flask import Flask, request, jsonify
import pickle
import re
import unicodedata
import numpy as np
from collections import Counter

app = Flask(__name__)


with open('model.pkl', 'rb') as model_file:
    model, vocab = pickle.load(model_file)


def preprocess_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def text_to_bow(text, vocab):
    word_counts = Counter(text.split())
    vector = np.zeros(len(vocab))
    for word, count in word_counts.items():
        if word in vocab:
            vector[vocab[word]] = count
    return vector.reshape(1, -1)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    content = data.get('content', '')

    if not content:
        return jsonify({'error': 'Content is required'}), 400

    preprocessed_content = preprocess_text(content)
    vectorized_content = text_to_bow(preprocessed_content, vocab)
    prediction = model.predict(vectorized_content)[0]

    return jsonify({'classification': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
