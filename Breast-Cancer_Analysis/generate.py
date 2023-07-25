import random
import csv
import numpy as np

def generate_random_samples(num_samples, num_features):
    random_samples = np.random.rand(num_samples, num_features) * 1000
    return random_samples

def save_samples_to_csv(samples, filename):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for sample in samples:
            formatted_sample = ["{:.3e}".format(value) for value in sample]
            writer.writerow(formatted_sample)

def main():
    print("Generating data...")
    num_samples = random.randint(1, 100)
    num_features = 30
    random_samples = generate_random_samples(num_samples, num_features)
    file_generate = "random_samples.csv"
    save_samples_to_csv(random_samples, file_generate)

    print(f"Generated data saved to '{file_generate}'.")

if __name__ == "__main__":
    main()
