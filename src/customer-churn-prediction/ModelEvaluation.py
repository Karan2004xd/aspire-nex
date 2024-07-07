from sklearn.metrics import confusion_matrix, accuracy_score

# Get the confusion matrix and accuracy_score
def get_auc(X_test, y_test, classifier):
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    score = accuracy_score(y_test, y_pred)

    return {
        'cm': cm,
        'score': score
    }
