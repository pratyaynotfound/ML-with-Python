# Naive Bayes Diabetes Classifier

This code demonstrates the implementation of a Naive Bayes classifier for diabetes classification. The dataset used contains features related to glucose and blood pressure, and the goal is to predict the presence or absence of diabetes based on these features.

## Dataset

The dataset used in this project is stored in the file [Naive-Bayes-Classification-Data.csv](link/to/dataset/Naive-Bayes-Classification-Data.csv). It contains the following columns:

- `glucose`: Glucose levels
- `bloodpressure`: Blood pressure
- `diabetes`: Target variable indicating the presence (1) or absence (0) of diabetes


## Code Overview

The code performs the following steps:

1. Loads the dataset using the `pandas` library.
2. Splits the dataset into features (`glucose` and `bloodpressure`) and the target variable (`diabetes`).
3. Splits the data into training and testing sets using the `train_test_split` function from `sklearn.model_selection`.
4. Creates a Naive Bayes classifier using the `GaussianNB` class from `sklearn.naive_bayes`.
5. Trains the classifier on the training set using the `fit` method.
6. Makes predictions on the test set using the `predict` method.
7. Calculates the training and testing accuracy using the `score` function from `sklearn.naive_bayes` and `accuracy_score` function from `sklearn.metrics`.
8. Displays the predicted values, training accuracy, and testing accuracy.
9. Visualizes the data by plotting a scatter plot of `glucose` against `bloodpressure` using `matplotlib.pyplot`.

## Dependencies

The code requires the following dependencies:

- `pandas`
- `sklearn`
- `matplotlib`

You can install the dependencies using `pip`:

```shell
pip install pandas scikit-learn matplotlib

```

## Usage
Place the dataset file Naive-Bayes-Classification-Data.csv in the same directory as the code.
Run the code in a Python environment with the required dependencies installed.
The code will load the dataset, train the Naive Bayes classifier, make predictions, and display the results.
Please note that the accuracy of the classifier may vary depending on the dataset and the characteristics of the data.


Feel free to modify the `readme.md` file according to your requirements or add any additional information you deem necessary.
