import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Load the Titanic training data from "train.csv" file
data = pd.read_csv("train.csv")

# Select relevant features and the target variable
features = ['Pclass', 'Age', 'Sex', 'Fare']
target = 'Survived'
x = data[features]
y = data[target]

# Convert categorical variables to numerical using one-hot encoding
x = pd.get_dummies(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a pipeline to handle missing values and train the SVM classifier
pipeline = make_pipeline(SimpleImputer(strategy='mean'), SVC(random_state=42))
pipeline.fit(x_train, y_train)

# Make predictions on the testing set
pred = pipeline.predict(x_test)
print("Predictions:", pred)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

# Calculate and print the confusion matrix
confusion = confusion_matrix(y_test, pred)
print("Confusion Matrix:")
print(confusion)

# Visualize the training data with two selected features
plt.scatter(x_train['Age'], x_train['Fare'], c=y_train, cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Titanic Training Data - Survived (1) vs. Not Survived (0)')
plt.show()
