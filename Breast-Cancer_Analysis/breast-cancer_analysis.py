import csv
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

print("Loading the csv...")
# Load the data
file_load = "new_samples.csv"
samples = []

with open(file_load, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        sample = [float(value) for value in row]
        samples.append(sample)
print("Loadiing done...")

print("ML started...")
# Start machine learning
data = load_breast_cancer()
x, y = data['data'], data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

predictions = knn.predict(samples)
predicted_class_names = [data['target_names'][prediction] for prediction in predictions]
print("ML complete.")

file_store = "store.csv"

with open(file_store, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Class Names"])
    for name in predicted_class_names:
        writer.writerow([name])

print(f"Data saved to '{file_store}'.")
