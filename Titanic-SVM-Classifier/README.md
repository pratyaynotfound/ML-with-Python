# Titanic SVM Classifier

This repository contains code for training a Support Vector Machine (SVM) classifier on the Titanic dataset to predict passenger survival.

## Prerequisites

- Python (>= 3.6)
- pandas
- scikit-learn
- matplotlib

## Usage

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your_username/titanic-svm-classifier.git
cd titanic-svm-classifier
```
1. Install the required dependencies:
```bash
   pip install pandas scikit-learn matplotlib
```
2. Download the Titanic dataset [train.csv] and place it in the same directory as the code.

3. Run the `titanic_surv.py` script to train the SVM classifier and make predictions on the test data.

4. The script will display the predicted values, accuracy score, and confusion matrix.

5. Additionally, it will show a scatter plot of the training data, highlighting passengers' ages and fares, with different colors for survived (1) and not survived (0).

## Results

The SVM classifier was trained and evaluated on the Titanic dataset for predicting passenger survival. Here are the results:

- Predictions: The SVM classifier made predictions on the testing set, indicating whether a passenger survived (1) or not survived (0).

```bash
Predictions: [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 1
 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
 0 1 0 1 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0]
```
- Accuracy: The accuracy score of the SVM classifier on the testing set is approximately 0.654, which means it correctly predicted survival outcomes for about 65.4% of passengers.

```bash
Accuracy: 0.6536312849162011
```
- Confusion Matrix: The confusion matrix provides a breakdown of correct and incorrect predictions made by the classifier.
```bash
Confusion Matrix:
[[99  6]
 [56 18]]
```
- True Positives (TP): 18 passengers who were correctly predicted as survivors.
- True Negatives (TN): 99 passengers who were correctly predicted as not survivors.
- False Positives (FP): 6 passengers who were incorrectly predicted as survivors.
- False Negatives (FN): 56 passengers who were incorrectly predicted as not survivors.
- The results indicate that the SVM classifier achieved reasonable accuracy in predicting passenger survival on the Titanic dataset. However, there is room for improvement, and additional exploration of the data and model tuning may lead to better performance.
- Visualization: The scatter plot below visualizes the training data with two selected features, 'Age' and 'Fare', showing different colors for passengers who survived (1) and those who did not survive (0).

[plt.img]

Please note that the provided accuracy score and confusion matrix are specific to the current run of the code. If you run the code again, you may get slightly different results due to the random nature of certain operations, such as data splitting and SVM fitting.

## Contributing

Contributions to this repository are welcome! If you have any improvements, bug fixes, or suggestions, please open an issue or submit a pull request.


Note: Make sure to update this README.md file with any changes to the code or additional features you add to the Titanic SVM Classifier repository.
