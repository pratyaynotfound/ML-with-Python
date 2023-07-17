import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Naive-Bayes-Classification-Data.csv")

# Split the features and target
x, y = data[['glucose', 'bloodpressure']], data['diabetes']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)

# Create a Naive Bayes classifier
nb = GaussianNB()

# Train the classifier
nb.fit(x_train, y_train)

# Make predictions on the test set
nb_pred = nb.predict(x_test)

# Print the predicted values
print(nb_pred)

# Calculate and print the training and testing accuracy
training_accuracy = nb.score(x_train, y_train) * 100
testing_accuracy = accuracy_score(y_test, nb_pred) * 100
print("Training Accuracy: ", training_accuracy, "%")
print("Testing Accuracy:", testing_accuracy, "%")

# Visualize the data
plt.scatter(x['glucose'], x['bloodpressure'])
plt.title("Diabetes Data")
plt.xlabel("Glucose")
plt.ylabel("Bloodpressure")
plt.show()
