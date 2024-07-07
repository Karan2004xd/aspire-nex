from sklearn.metrics import precision_recall_curve, auc

# Calculate the area under the curve, for the imbalanced data
def get_auc(X_test, y_test, classifier):
    y_pred_proba = classifier.predict_proba(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
    auprc = auc(recall, precision)
    return {
        'precision': precision,
        'recall': recall,
        'auprc': auprc
    }
