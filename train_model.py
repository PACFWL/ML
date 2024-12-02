import pandas as pd
import re
import unicodedata
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
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


data = data.dropna(subset=['content', 'classification']) 
data['content'] = data['content'].apply(str).apply(preprocess_text)


label_encoder = LabelEncoder()
data['classification'] = label_encoder.fit_transform(data['classification'])

X = data['content']
y = data['classification']


vocab = build_vocabulary(X)
X_bow = texts_to_bow(X, vocab)


model = LogisticRegression(max_iter=1000, class_weight='balanced')


scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

cv_results = {}
for metric, scorer in scoring.items():
    scores = cross_val_score(model, X_bow, y, cv=5, scoring=scorer)
    cv_results[metric] = scores

print("Resultados da validação cruzada:")
for metric, scores in cv_results.items():
    print(f"{metric.capitalize()}: Média = {np.mean(scores):.4f}, Desvio padrão = {np.std(scores):.4f}")


model.fit(X_bow, y)


with open('model.pkl', 'wb') as model_file:
    pickle.dump((model, vocab, label_encoder), model_file)

print('Modelo treinado e salvo com sucesso!')
