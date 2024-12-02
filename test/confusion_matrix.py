import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


y_pred = model.predict(X_test)


conf_matrix = confusion_matrix(y_test, y_pred)


disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='viridis', values_format='d')
