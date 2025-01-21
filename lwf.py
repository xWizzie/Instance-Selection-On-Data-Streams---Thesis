import numpy as np

from Utils import save_results_to_excel, test_then_train
from data import list_and_select_dat_files

class LocallyWeightedForgetting:
    def __init__(self, k=3, decay_rate=0.46, deletion_threshold=0.04):
        """
        Initialize the LWF algorithm.

        Parameters:
        k: int - Number of nearest neighbors to consider.
        decay_rate: float - Maximum decay rate.
        deletion_threshold: float - Threshold below which exemplars are removed.
        """
        self.k = k
        self.decay_rate = decay_rate
        self.deletion_threshold = deletion_threshold
        self.learning_set = []  # Stores exemplars as (x, y, weight).

    def get_active_set_length(self):
        return len(self.learning_set)
    
    def fit(self, data, labels):
        """
        Fit the model to the provided data and labels.

        Parameters:
        data: np.ndarray - The input data, of shape (n_samples, n_features).
        labels: np.ndarray - The labels corresponding to the data.
        """
        for x, y in zip(data, labels):
            self._update_learning_set(x, y)

    def predict(self, data):
        """
        Predict labels for the given data.

        Parameters:
        data: np.ndarray - Input data of shape (n_samples, n_features).

        Returns:
        np.ndarray - Predicted labels.
        """
        predictions = []
        for x in data:
            neighbors = self._get_neighbors(x)
            if neighbors:
                # Weighted majority voting
                labels, weights = zip(*[(neighbor[1], neighbor[2]) for neighbor in neighbors])
                predictions.append(np.argmax(np.bincount(labels, weights=weights)))
            else:
                predictions.append(-1)  # Unknown label if no neighbors
        return np.array(predictions)

    def _update_learning_set(self, x, y):
        """
        Update the learning set with a new exemplar.

        Parameters:
        x: np.ndarray - The input vector.
        y: int - The label of the input vector.
        """
        neighbors = self._get_neighbors(x)
        # Update weights of neighbors
        for neighbor in neighbors:
            neighbor[2] *= self._compute_decay(x, neighbor[0])

        # Remove neighbors below threshold
        self.learning_set = [ex for ex in self.learning_set if ex[2] >= self.deletion_threshold]

        # Add the new exemplar
        self.learning_set.append([x, y, 1.0])

    def _get_neighbors(self, x):
        """
        Retrieve k-nearest neighbors for a given input vector.

        Parameters:
        x: np.ndarray - The input vector.

        Returns:
        list - The k-nearest neighbors with their weights.
        """
        if not self.learning_set:
            return []

        distances = [(ex, np.linalg.norm(x - ex[0])) for ex in self.learning_set]
        distances.sort(key=lambda d: d[1])
        return [d[0] for d in distances[:self.k]]

    def _compute_decay(self, x, neighbor_x):
        """
        Compute the decay factor for a neighbor based on distance.

        Parameters:
        x: np.ndarray - The input vector.
        neighbor_x: np.ndarray - The neighbor's vector.

        Returns:
        float - The decay factor.
        """
        distance = np.linalg.norm(x - neighbor_x)
        return max(self.decay_rate - distance**2, self.deletion_threshold)

# Adjusted test_then_train function for LWF

# for i in range(1,14):
# Example usage
data, labels, batch_size = list_and_select_dat_files('Datasets',4)

lwf = LocallyWeightedForgetting()
metrics_table, cpu_time_table = test_then_train(lwf, data, labels, batch_size=batch_size)
# Print results
for entry in metrics_table:
    print(entry)  

# Save results to Excel
save_results_to_excel(metrics_table, "LWF",cpu_time_metrics=cpu_time_table)