from flask import Flask, request, jsonify
import pickle
import re
import unicodedata
import csv
import numpy as np
from collections import Counter
from nltk.corpus import stopwords 
import nltk
nltk.download('stopwords')

app = Flask(__name__)


with open('model.pkl', 'rb') as model_file:
    model, vocab = pickle.load(model_file)


def load_emoji_map(filepath):
    emoji_map = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            emoji_map[row['emoji']] = row['sentimento']
    return emoji_map


def load_giria_map(filepath):
    giria_map = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            giria_map[row['giria']] = row['formal']
    return giria_map


def replace_emojis(text, emoji_map):
    for emoji, description in emoji_map.items():
        text = text.replace(emoji, description)
    return text


def replace_girias(text, giria_map):
    words = text.split()
    replaced = []
    for word in words:
        replaced.append(giria_map.get(word, word))
    return ' '.join(replaced)


def normalize_repeated_characters(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)


def remove_stopwords(text):
    stop_words = set(stopwords.words('portuguese'))
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)


def clean_non_textual_elements(text):
    text = re.sub(r'http\S+|www.\S+', '<url>', text)  
    text = re.sub(r'@\w+', '<mention>', text)         
    text = re.sub(r'#\w+', '<hashtag>', text)        
    return text

# Carregar mapeamentos de emojis e gírias
emoji_map = load_emoji_map('emoji_map.csv')
giria_map = load_giria_map('giria_map.csv')

# Função de pré-processamento do texto
def preprocess_text(text):
    # Substituir emojis
    text = replace_emojis(text, emoji_map)
    # Substituir gírias
    text = replace_girias(text, giria_map)
    # Normalizar palavras com letras repetidas
    text = normalize_repeated_characters(text)
    # Remover elementos não textuais
    text = clean_non_textual_elements(text)
    # Normalizar texto e remover caracteres desnecessários
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    # Remover stopwords
    text = remove_stopwords(text)
    return text

# Converter texto em vetor Bag-of-Words
def text_to_bow(text, vocab):
    word_counts = Counter(text.split())
    vector = np.zeros(len(vocab))
    for word, count in word_counts.items():
        if word in vocab:
            vector[vocab[word]] = count
    return vector.reshape(1, -1)

# Endpoint para classificação
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
