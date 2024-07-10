About
-----
Contains two machine learning models implemented in python.
- Model one is a credit card fault detector
- Model two is a customer churn predictor

Structure of repository
-----------------------

- **notebooks**: Contains the google colaborotory files for the both task implementation
- **src**: Contains the implementation of both models

Steps to run the models
-----------------------

- **Using the notebooks**
    - You can simply use the colab notebooks to test out models in that environment

- **Using the python scripts**
    - First you will need to clone the repository
    ``````bash
    git clone https://github.com/Karan2004xd/aspire-nex.git
    ``````
    - Make sure to have the dependencies installed before running the script (the list of dependencies can be found in requirements.txt)

    - **Unzip** the file datasets.zip, to get the datasets

    - Navigate to **src** directory, here you will find the directory for both models, 
        inside each model there is a **run.py** file using which you can simply run the model

    - **NOTE**: Make sure to navigate till the directory, where run.py is in the same directory, reason for that is that the script uses relative path for finding the dataset from the datasets directory
