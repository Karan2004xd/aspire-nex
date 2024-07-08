from DataPreprocessing import data_preprocessing
from TrainingModels import *
from ModelEvaluation import get_auc

def get_all_classifiers_dict(X_train, y_train):
    return {
        'gradient_boosting_clasifier': gradient_boosting_clasifier(X_train, y_train),
        'random_forest_classifier': random_forest_classifier(X_train, y_train),
        'Logistic_regression_classifier': logistic_regression_classifier(X_train, y_train),
        'decision_tree_classifier': decision_tree_classifier(X_train, y_train)
    }

def display_results(result):
    cm, score = result['cm'], result['score']
    print("\nConfusion Matrix:")
    print(cm)

    print()

    print(f"Accuracy Score: {score}")

def setup_results(classifier, X_test, y_test):
    if (classifier == None):
        print('Invalid argument')
        return 

    if (type(classifier) == dict):
        print('\nResults from top to down (models trained) respectively')
        for clss in classifier.values():
            result = get_auc(X_test, y_test, clss)
            display_results(result)
    else:
        result = get_auc(X_test, y_test, classifier)
        display_results(result)

def run():
    model_data = data_preprocessing()
    X_train, X_test, y_train, y_test = model_data['X_train'], model_data['X_test'], model_data['y_train'], model_data['y_test']

    print('Choose the algorithm to train model with get results for the same::')
    print('(1) Gradinet Boosting')
    print('(2) Random Forest')
    print('(3) Logistic Regression')
    print('(4) Decision Tree')
    print('(5) All (can take longer time to finish)')
    
    choice = eval(input('\n\rYour choice (1, 2, 3, 4, 5) : '))
    classifier = None

    if (choice == 1):
        classifier = gradient_boosting_clasifier(X_train, y_train)
    elif (choice == 2):
        classifier = random_forest_classifier(X_train, y_train)
    elif (choice == 3):
        classifier = logistic_regression_classifier(X_train, y_train)
    elif (choice == 4):
        classifier = decision_tree_classifier(X_train, y_train)
    else:
        classifier = get_all_classifiers_dict(X_train, y_train)

    setup_results(classifier, X_test, y_test)

run()
