# Support Vector Machine Classifier with PCA Visualization for Iris Dataset

This repository contains Python code that demonstrates the implementation of a Support Vector Machine (SVM) classifier on the famous Iris dataset. The code also includes a visualization of the data after reducing its dimensionality using Principal Component Analysis (PCA).

## Requirements

The following Python libraries are required to run the code:

- scikit-learn
- matplotlib

You can install the required libraries using the following command:

```bash
pip install scikit-learn matplotlib

```


## Code Description

The code can be found in the `svm_iris_pca.py` file. It performs the following tasks:

1. Loads the Iris dataset and splits it into features and labels.
2. Divides the dataset into training and testing sets.
3. Initializes and trains a Support Vector Machine (SVM) classifier on the training set.
4. Makes predictions on the test set using the trained SVM model.
5. Calculates and prints the accuracy of the model on both the training and test sets.
6. Prints the confusion matrix to assess the classifier's performance.
7. Performs Principal Component Analysis (PCA) to reduce the feature dimensionality to 2 columns.
8. Plots the reduced training set in a 2D scatter plot, with different colors for different classes.

## How to Run

1. Ensure you have Python and the required libraries installed.
2. Clone this repository to your local machine.
3. Navigate to the repository folder.
4. Run the `SVM_Iris.py` script using the following command:

```bash
python SVM_Iris.py

```


## Results

The code provides insights into the performance of the SVM classifier on the Iris dataset and its visualization in a 2D space after applying PCA.

Feel free to explore and modify the code to experiment with different classifiers or other machine learning techniques.

Enjoy!
