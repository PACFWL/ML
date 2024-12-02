import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]


disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_normalized, display_labels=label_encoder.classes_)
disp.plot(cmap='viridis', values_format='.2f')
plt.title("Matriz de Confusão Normalizada")
plt.show()
