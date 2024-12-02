import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


with open('model.pkl', 'rb') as model_file:
    model, vocab, label_encoder = pickle.load(model_file)


data = pd.read_csv('reviews.csv')
data = data.dropna(subset=['content', 'classification'])
data['content'] = data['content'].apply(str).apply(preprocess_text)
data['classification'] = label_encoder.transform(data['classification'])


X_bow = texts_to_bow(data['content'], vocab)
y = data['classification']


X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.3, random_state=42)


y_proba = model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr') 
print(f"ROC-AUC (multi-class): {roc_auc:.4f}")
