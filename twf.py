import numpy as np
from sklearn.metrics import accuracy_score
from Utils import save_results_to_excel, test_then_train
from data import list_and_select_dat_files
from scipy.spatial import distance
import time

class TWF:
    def __init__(self, forgetting_rate=0.9, threshold=0.02):
        self.forgetting_rate = forgetting_rate
        self.threshold = threshold
        self.learning_set = []  # List to store exemplars (points, labels, weights)

    def get_active_set_length(self):
        return len(self.learning_set)
    
    @property
    def condensing_set(self):
        """Return the current learning set (condensing set)."""
        return self.learning_set

    def add_batch(self, batch_points, batch_labels):
        """
        Add multiple data points and their labels to the learning set in a batch.

        :param batch_points: A numpy array of data points to add.
        :param batch_labels: A numpy array or list of corresponding labels.
        """
        if len(batch_points) != len(batch_labels):
            raise ValueError("The number of points and labels must match.")
        
        for point, label in zip(batch_points, batch_labels):
            self.add_data_point(point, label)

    def predict(self, test_points, k=1):
        """
        Predict the label for a given test point using the k-Nearest Neighbors approach.

        :param test_points: The test points to classify (numpy array).
        :param k: The number of nearest neighbors to consider (default is 3).
        :return: The predicted labels.
        """
        if not self.learning_set:
            raise ValueError("The learning set is empty. Add data points before predicting.")
        
        predictions = []
        for test_point in test_points:
            # Compute distances from the test point to all points in the learning set
            distances = []
            for point, label, weight in self.learning_set:
                dist = distance.euclidean(test_point, point)
                distances.append((dist, label, weight))
            
            # Sort by distance and select the k closest points
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:k]
            
            # Perform a weighted vote based on the k nearest neighbors
            label_votes = {}
            for _, label, weight in k_nearest:
                if label in label_votes:
                    label_votes[label] += weight
                else:
                    label_votes[label] = weight
            
            # Determine the label with the highest weighted vote
            predicted_label = max(label_votes, key=label_votes.get)
            predictions.append(predicted_label)
        
        return np.array(predictions)

    def add_data_point(self, point, label):
        # Add new exemplar with initial weight = 1.0
        self.learning_set.append((point, label, 1.0))
        
        # Update weights for all exemplars in the learning set
        updated_learning_set = []
        for data_point, data_label, weight in self.learning_set:
            # Apply decay
            new_weight = weight * self.forgetting_rate
            if new_weight >= self.threshold:
                updated_learning_set.append((data_point, data_label, new_weight))
        
        # Replace learning set with updated set
        self.learning_set = updated_learning_set

    def fit(self, batch_points, batch_labels):
        """
        Train the model on a batch of points and labels.
        """
        self.add_batch(batch_points, batch_labels)


def find_best_combination():
    # Define the ranges for forgetting rate and threshold
    forgetting_rates = np.arange(0.5, 0.95, 0.05)  # Forgetting rates from 0.5 to 1.0 in steps of 0.05
    thresholds = np.arange(0.01, 4.01, 0.01)       # Thresholds from 0.01 to 4.0 in steps of 0.01


    def run_tw_for_combination(data, labels, forgetting_rate, threshold, batch_size=100):
        twf = TWF(forgetting_rate=forgetting_rate, threshold=threshold)
        accuracies = []  # List to store accuracies
        learning_set_sizes = []  # List to store the learning set sizes
        
        for i in range(0, len(data), batch_size):
            # Get current batch
            batch_points = data[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            # Test accuracy before adding the batch
            correct_predictions = 0
            if twf.learning_set:  # Ensure the learning set is not empty
                for test_point, true_label in zip(batch_points, batch_labels):
                    predicted_label = twf.predict(test_point, k=3)
                    if predicted_label == true_label:
                        correct_predictions += 1
                accuracy = correct_predictions / len(batch_points)
                accuracies.append(accuracy)
            else:
                accuracies.append(None)
            
            # Add batch to the learning set
            twf.add_batch(batch_points, batch_labels)
            learning_set_sizes.append(len(twf.learning_set))
        # Return the mean accuracy ignoring None values
        mean_accuracy = np.mean([acc for acc in accuracies if acc is not None])
        
        return mean_accuracy, learning_set_sizes[-1]  # Return mean accuracy and final learning set size

    # Run for all combinations
    best_combination = None
    best_accuracy = -1

    for forgetting_rate in forgetting_rates:
        for threshold in thresholds:
            print(f"Testing with forgetting_rate={forgetting_rate}, threshold={threshold}")
            mean_accuracy, final_learning_size = run_tw_for_combination(data, labels, forgetting_rate, threshold)
            
            print(f"Mean Accuracy: {mean_accuracy:.4f}, Final Learning Set Size: {final_learning_size}")
            
            # Update best combination if this is the best so far
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_combination = (forgetting_rate, threshold)

    print(f"Best Combination: Forgetting Rate={best_combination[0]}, Threshold={best_combination[1]}, Accuracy={best_accuracy:.4f}")


for i in range(1, 15):
    print(f"Starting dataset {i}")
    # Example usage
    data, labels, batch_size = list_and_select_dat_files('Datasets', i)
    print(f"Loaded dataset {i} with {len(data)} samples.")
    
    model = TWF()
    print("Starting AIB fitting...")
    model.fit(data, labels)
    print("Finished AIB fitting.")

    print("Starting test-then-train evaluation...")
    results, cpu_time_table = test_then_train(
        model, data, labels, batch_size=batch_size)
    print("Finished evaluation.")
    
    for entry in results:
        print(entry)

    save_results_to_excel(results, "TWF", cpu_time_metrics=cpu_time_table)