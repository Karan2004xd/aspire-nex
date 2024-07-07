# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def data_preprocessing():
    print('Pre-Processing the data...')
    
    # import the dataset
    dataset = pd.read_csv('../../dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Seperate the dependent and independent variables
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    # Clean the missing data
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

    # Encoding multiple categorical data
    ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [6, 7, 8, 9, 10, 11, 12, 13, 14, 16])], remainder = 'passthrough')
    X = np.array(ct.fit_transform(X))

    # Split the data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train[:, -2:] = sc.fit_transform(X_train[:, -2:])
    X_test[:, -2:] = sc.transform(X_test[:, -2:])

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
