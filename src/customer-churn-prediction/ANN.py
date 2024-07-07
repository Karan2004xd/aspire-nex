# Importing libraries and dataset
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

def data_preprocessing():

    # importing the dataset
    dataset = pd.read_csv('../../dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Seperating the independent and dependent variables
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    # Cleaning missing data
    imputer = SimpleImputer(missing_values=' ', strategy="constant", fill_value = '0')
    imputer.fit(X[:, -1:])

    X[:, -1:] = imputer.transform(X[:, -1:])
    X[:, -1] = X[:, -1].astype(float)

    # Encoding the single categorical data
    le = LabelEncoder()
    label_category = [2, 3, 5, 15]

    X[:, 0] = le.fit_transform(X[:, 0])

    le.fit(X[:, 2])
    for i in label_category:
        X[:, i] = le.transform(X[:, i])

    y = le.transform(y)

    # Encoding the multiple categorical data
    ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [6, 7, 8, 9, 10, 11, 12, 13, 14, 16])], remainder = 'passthrough')
    X = np.array(ct.fit_transform(X))

    # Spliting the data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

def train_and_compile(X_train, y_train):
    ann = tf.keras.models.Sequential()

    ann.add(tf.keras.layers.Dense(units = 10, activation = 'relu'))
    ann.add(tf.keras.layers.Dense(units = 10, activation = 'relu'))
    ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
    return ann

def get_result(ann, X_test, y_test):
    y_pred = ann.predict(X_test)
    y_pred = y_pred > 0.5

    cm = confusion_matrix(y_test, y_pred)
    score = accuracy_score(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    print()

    print(f"Accuracy Score: {score}")

def predict_using_ann():
    model_data = data_preprocessing()
    X_train, X_test, y_train, y_test = model_data['X_train'], model_data['X_test'], model_data['y_train'], model_data['y_test']

    ann = train_and_compile(X_train, y_train)
    get_result(ann, X_test, y_test)
