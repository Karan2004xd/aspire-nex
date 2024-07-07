from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def k_n_classifier(X_train, y_train):
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    print("Training the model using the K-Nearest Neighbour Algorithm...")
    classifier.fit(X_train, y_train)
    return classifier

def random_forest_classifier(X_train, y_train):
    classifier = RandomForestClassifier(random_state=0, n_estimators = 100, criterion = 'entropy')
    print("Training the model using the Random Forest Algorithm...")

    classifier.fit(X_train, y_train)
    return classifier

def logistic_regression_classifier(X_train, y_train):
    classifier = LogisticRegression(class_weight = 'balanced')
    print("Training the model using the Logistic Regression Algorithm...")

    classifier.fit(X_train, y_train)
    return classifier

def decision_tree_classifier(X_train, y_train):
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    print("Training the model using the Decision Tree Algorithm...")

    classifier.fit(X_train, y_train)
    return classifier
