import pandas as pd
import re
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import nltk


nltk.download('stopwords')
nltk.download('rslp')


stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()


def preprocess_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)


def build_vocabulary(texts):
    vocab = set()
    for text in texts:
        vocab.update(text.split())
    return {word: idx for idx, word in enumerate(sorted(vocab))}


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


data = pd.read_csv('reviews.csv')
data['content'] = data['content'].apply(preprocess_text)

X = data['content']
y = data['classification']


vocab = build_vocabulary(X)
X_bow = texts_to_bow(X, vocab)


X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.25, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


conf_matrix = confusion_matrix(y_test, y_pred)


sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()


with open('model.pkl', 'wb') as model_file:
    pickle.dump((model, vocab), model_file)

print('Modelo treinado, gráfico exibido e modelo salvo com sucesso!')
