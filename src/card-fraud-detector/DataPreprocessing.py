# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_preprocessing():
    print('Pre-Processing the data...')
    # import the dataset
    dataset = pd.read_csv('../../datasets/creditcard.csv')

    # Seperate the dependent and independent variables
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Split the data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train[:, :1] = sc.fit_transform(X_train[:, :1])
    X_train[:, -1:] = sc.transform(X_train[:, -1:])

    X_test[:, :1] = sc.transform(X_test[:, :1])
    X_test[:, -1:] = sc.transform(X_test[:, -1:])

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
