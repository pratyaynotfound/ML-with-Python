import random
import csv
import numpy as np

print("Generating data...")
#generate data
num_samples = random.randint(1,100)
random_samples = np.random.rand(num_samples, 30) * 1000
formatted_samples = [["{:.3e}".format(value) for value in sample] for sample in random_samples]
file_generate = "random_samples.csv"
with open("new_samples.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(formatted_samples)

print(f"Generated data saved to '{file_generate}'.")