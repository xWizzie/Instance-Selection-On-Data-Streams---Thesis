import numpy as np
from Utils import save_results_to_excel, test_then_train
from data import list_and_select_dat_files
from sklearn.metrics import accuracy_score


class AIB:
    def __init__(self):
        # List to store the condensing set (prototypes and their details)
        self.condensing_set = []

    def get_active_set_length(self):
        return len(self.condensing_set)

    def fit(self, data, labels):
        print("Began Fitting")
        """Fit the AIB model to the given data and labels."""

        # If condensing set is empty, initialize with the first data point
        if not self.condensing_set:
            self.condensing_set.append({
                'prototype': data[0],
                'weight': 1,
                'label': labels[0]
            })
            data = data[1:]  # Remove the first element
            labels = labels[1:]

        # Pre-extract prototypes and labels to avoid repeated access
        prototypes = np.array([entry['prototype'] for entry in self.condensing_set])
        weights = np.array([entry['weight'] for entry in self.condensing_set])
        prototype_labels = np.array([entry['label'] for entry in self.condensing_set])

        for i, (x, label) in enumerate(zip(data, labels)):
            if i % 100 == 0:  # Add periodic progress tracking
                print(f"Processing sample {i}/{len(data)}")

            # Calculate distances from the current data point to all prototypes
            distances = np.linalg.norm(prototypes - x, axis=1)
            nearest_index = np.argmin(distances)

            # If the nearest prototype misclassifies the current point
            if prototype_labels[nearest_index] != label:
                # Add the current point as a new prototype
                self.condensing_set.append({
                    'prototype': x,
                    'weight': 1,
                    'label': label
                })

                # Update the prototypes, weights, and labels arrays
                prototypes = np.append(prototypes, [x], axis=0)
                weights = np.append(weights, 1)
                prototype_labels = np.append(prototype_labels, label)
            else:
                # Update the nearest prototype using the weighted mean
                weight = weights[nearest_index]
                prototypes[nearest_index] = (
                    (weight * prototypes[nearest_index] + x) /
                    (weight + 1)
                )
                weights[nearest_index] += 1

        # At the end, reassign the optimized condensing set
        self.condensing_set = [
            {'prototype': prototypes[i], 'weight': weights[i], 'label': prototype_labels[i]}
            for i in range(len(prototypes))
        ]


    def predict(self, data):
        """Predict the labels for the given data."""
        prototypes = np.array([entry['prototype'] for entry in self.condensing_set])
        labels = np.array([entry['label'] for entry in self.condensing_set])
        
        predictions = []
        for i, x in enumerate(data):
            if i % 100 == 0:  # Add periodic progress tracking
                print(f"Predicting sample {i}/{len(data)}")

            # Compute distances between x and all prototypes in a vectorized manner
            distances = np.linalg.norm(prototypes - x, axis=1)
            nearest_index = np.argmin(distances)
            predictions.append(labels[nearest_index])
        return np.array(predictions)


for i in range(1, 15):
    print(f"Starting dataset {i}")
    # Example usage
    data, labels, batch_size = list_and_select_dat_files('Datasets', i)
    print(f"Loaded dataset {i} with {len(data)} samples.")
    
    model = AIB()
    print("Starting AIB fitting...")
    model.fit(data, labels)
    print("Finished AIB fitting.")

    print("Starting test-then-train evaluation...")
    results, cpu_time_table = test_then_train(
        model, data, labels, batch_size=batch_size)
    print("Finished evaluation.")
    
    for entry in results:
        print(entry)

    save_results_to_excel(results, "AIB2", cpu_time_metrics=cpu_time_table)
