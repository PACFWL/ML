import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter


def preprocess_text(text):
  
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

   
    words = text.split()
    words = [word for word in words if word not in stop_words]

  
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)


def texts_to_bow(texts, vocab):
    vectors = []
    for text in texts:
        word_counts = Counter(text.split())
        vector = np.zeros(len(vocab))
        for word, count in word_counts.items():
            if word in vocab:
                vector[vocab[word]] = count
        vectors.append(vector)
    return np.array(vectors)


with open('model.pkl', 'rb') as model_file:
    model, vocab, label_encoder = pickle.load(model_file)


data = pd.read_csv('reviews.csv')
data = data.dropna(subset=['content', 'classification'])
data['content'] = data['content'].apply(str).apply(preprocess_text)


X_test = data['content']
y_test = label_encoder.transform(data['classification'])


X_test_bow = texts_to_bow(X_test, vocab)


y_pred = model.predict(X_test_bow)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Resultados do teste:")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Revocação: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
