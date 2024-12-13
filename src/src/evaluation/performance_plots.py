import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix", filename=None):
    """
    Plot the confusion matrix for predictions and save it as an image.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - classes: List of class names.
    - title: Title of the confusion matrix plot.
    - filename: Filename to save the image. (Optional)
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved as {filename}")

    plt.show()
