from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd


with open('model.pkl', 'rb') as model_file:
    model, vocab, label_encoder = pickle.load(model_file)


data = pd.read_csv('reviews.csv')
data = data.dropna(subset=['content', 'classification'])
data['content'] = data['content'].apply(str).apply(preprocess_text)
data['classification'] = label_encoder.transform(data['classification'])


X_bow = texts_to_bow(data['content'], vocab)
y = data['classification']


train_sizes, train_scores, test_scores = learning_curve(
    model, X_bow, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)


train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)


plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, label="Treinamento", marker='o')
plt.plot(train_sizes, test_scores_mean, label="Validação", marker='o')
plt.xlabel("Tamanho do Conjunto de Treinamento")
plt.ylabel("Acurácia")
plt.title("Curva de Aprendizado")
plt.legend()
plt.grid()
plt.show()
