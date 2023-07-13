import pickle
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Download and load the dataset
data_path = 'cifar-10-batches-py'

# Load the dataset using pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

# Load data batch
data = unpickle(data_path + '/data_batch_1')

# Load batches.meta for label names
label_data = unpickle(data_path + '/batches.meta')
label_names = [label.decode('utf-8') for label in label_data[b'label_names']]

# Normalize the pixel values
features = data[b'data'] / 255.0
labels = data[b'labels']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Random Forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

pred = rf.predict(x_test)
score = rf.score(x_train, y_train)
accu = accuracy_score(y_test, pred)

# Set print options to display all elements in pred
np.set_printoptions(threshold=sys.maxsize)

# Print the predictions
print("Predictions:", pred)

# Print accuracy and score
print("Accuracy:", accu)
print("Score:", score)

# Count predictions by label
counts = np.zeros(10)
for i in pred:
    counts[pred[i]] += 1

# Print counts
print("Counts by Label:", counts)

# Create a bar graph for the counts
plt.bar(label_names, counts)
plt.xlabel('Label')
plt.ylabel('Counts')
plt.title('Counting Predictions by Label')
plt.show()
