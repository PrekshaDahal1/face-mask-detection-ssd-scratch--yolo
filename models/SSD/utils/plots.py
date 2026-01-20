import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(
        cmap=plt.cm.Blues,   # light blue pastel theme
        colorbar=True,
        values_format="d"
    )

    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_curve(x, y, xlabel, ylabel, save_path):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()
