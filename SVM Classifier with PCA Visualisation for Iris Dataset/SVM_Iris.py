# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Loading the Iris dataset and splitting it into features (x) and labels (y)
x, y = load_iris(return_X_y=True)

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initializing and training a Support Vector Machine (SVM) classifier
svm = SVC()
svm.fit(x_train, y_train)

# Making predictions on the test set using the trained SVM model
pred = svm.predict(x_test)
print(pred)

# Calculating and printing the accuracy of the model on the training set
print(svm.score(x_train, y_train))

# Calculating and printing the accuracy of the model on the test set using accuracy_score function
print(accuracy_score(y_test, pred))

# Printing the confusion matrix to evaluate the performance of the model
print(confusion_matrix(y_test, pred))

# Performing Principal Component Analysis (PCA) to reduce the feature dimensionality to 2 columns
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# Plotting the reduced training set in a 2D scatter plot, using different colors for different classes
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.show()
