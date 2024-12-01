from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Carregar modelo e vetorizador
with open('model.pkl', 'rb') as model_file:
    model, vectorizer = pickle.load(model_file)

# Função de pré-processamento
def preprocess_text(text):
    import re
    import unicodedata
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    content = data.get('content', '')

    if not content:
        return jsonify({'error': 'Content is required'}), 400

    preprocessed_content = preprocess_text(content)
    vectorized_content = vectorizer.transform([preprocessed_content])
    prediction = model.predict(vectorized_content)[0]

    return jsonify({'classification': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
