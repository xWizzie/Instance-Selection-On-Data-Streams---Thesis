import numpy as np
from Utils import save_results_to_excel, test_then_train
from data import list_and_select_dat_files

import numpy as np

class PECS:
    def __init__(self, k=5, p_min=0.6, p_max=0.4, shift_register_length=20):
        """
        Initialize the PECS algorithm parameters.

        Parameters:
        - k: Number of nearest neighbors
        - p_min: Minimum acceptance probability for activation
        - p_max: Maximum disagreement probability for deactivation
        - shift_register_length: Length of the agreement shift register
        """
        self.k = k
        self.p_min = p_min
        self.p_max = p_max
        self.shift_register_length = shift_register_length
        self.active_set = []
        self.inactive_set = []

    def fit(self, data, labels):
        """
        Fit the initial dataset.

        Parameters:
        - data: Array-like, shape (n_samples, n_features)
        - labels: Array-like, shape (n_samples,)
        """
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.active_set = [{'data': x, 'label': y, 'history': [
            1] * self.shift_register_length} for x, y in zip(data, labels)]

    def fit_batch(self, batch_data, batch_labels):
        """
        Train the model in batches.

        Parameters:
        - batch_data: Array-like, shape (batch_size, n_features)
        - batch_labels: Array-like, shape (batch_size,)
        """
        for data, label in zip(batch_data, batch_labels):
            self.update(data, label)

    def predict(self, queries):
        """
        Predict the labels for a batch of query points.

        Parameters:
        - queries: Array-like, shape (n_samples, n_features)

        Returns:
        - Predicted labels: Array-like, shape (n_samples,)
        """
        predictions = []
        for query in queries:
            # Compute distances to active set points
            distances = [np.linalg.norm(query - a['data']) for a in self.active_set]
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.active_set[i]['label'] for i in nearest_indices]

            # Majority vote
            predicted_label = max(set(nearest_labels), key=nearest_labels.count)
            predictions.append(predicted_label)

        return np.array(predictions)  # Ensure output is array-like


    def update(self, new_data, new_label):
        """
        Update the active and inactive sets with a new observation.

        Parameters:
        - new_data: Array-like, shape (n_features,)
        - new_label: Label associated with the new data point
        """
        # Add new data to active set
        new_entry = {'data': np.array(new_data), 'label': new_label, 'history': [
            1] * self.shift_register_length}
        self.active_set.append(new_entry)

        # Update neighbors' agreement history
        distances = [np.linalg.norm(new_data - a['data'])
                     for a in self.active_set]
        nearest_indices = np.argsort(distances)[:self.k]

        for idx in nearest_indices:
            entry = self.active_set[idx]
            agreement = int(entry['label'] == new_label)
            entry['history'].pop(0)  # Remove oldest record
            entry['history'].append(agreement)  # Add newest agreement

            agreement_ratio = sum(entry['history']) / self.shift_register_length

            if agreement_ratio < self.p_max:
                # Move to inactive set
                self.inactive_set.append(entry)
                self.active_set = [e for e in self.active_set if not np.array_equal(
                    e['data'], entry['data'])]
            elif agreement_ratio > self.p_min:
                # Re-activate from inactive set if needed
                if any(np.array_equal(e['data'], entry['data']) for e in self.inactive_set):
                    self.inactive_set = [
                        e for e in self.inactive_set if not np.array_equal(e['data'], entry['data'])]
                    self.active_set.append(entry)

    def get_active_set_length(self):
        """
        Return the length of the active set.
        """
        return len(self.active_set)


for i in range(1, 15):
    print(f"Starting dataset {i}")
    # Example usage
    data, labels, batch_size = list_and_select_dat_files('Datasets', i)
    print(f"Loaded dataset {i} with {len(data)} samples.")
    
    model = PECS(k=1)
    print("Starting AIB fitting...")
    model.fit(data, labels)
    print("Finished AIB fitting.")

    print("Starting test-then-train evaluation...")
    results, cpu_time_table = test_then_train(
        model, data, labels, batch_size=batch_size)
    print("Finished evaluation.")
    
    for entry in results:
        print(entry)

    save_results_to_excel(results, "PECS", cpu_time_metrics=cpu_time_table)