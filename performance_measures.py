import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

"""
Set of functions that gets performance scores and plots ROC-curve.
Negative class is Acne Vulgaris, positive is Acne Rosacea.
"""


def get_sensitivity(y_true, y_pred, labels=None):
    # sensitivity is TPR
    cm = confusion_matrix(y_true, y_pred, labels)
    return cm[1,1]/(cm[1,1]+cm[1,0])

def get_specificity(y_true, y_pred, labels=None):
    # specificity is TNR
    cm = confusion_matrix(y_true, y_pred, labels)
    return cm[0,0] / (cm[0,0] + cm[0,1])

# Roc needs scores instead of predictions
def get_roc_auc(y_true, y_probs, plot=True):
    # get AUC-score
    roc_auc = roc_auc_score(y_true, y_probs)

    if plot:
        # Compute ROC-curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    return roc_auc
