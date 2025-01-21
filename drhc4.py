import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

from Utils import save_results_to_excel, test_then_train
from data import list_and_select_dat_files

class DRHC:
    def __init__(self, homogeneity_threshold=0.9, max_clusters=None):
        self.condensing_set = []  # Stores prototypes as (centroid, label, weight)
        self.homogeneity_threshold = homogeneity_threshold
        self.max_clusters = max_clusters

    def fit(self, data, labels):
        """
        Fit the model using recursive clustering to build the condensing set.
        """
        self._recursive_cluster(data, labels)

    def update_with_new_data(self, new_data, new_labels):
        """
        Update the condensing set with new data using weighted clustering.
        """
        if not self.condensing_set:
            # Initialize condensing set with the first batch
            for point, label in zip(new_data, new_labels):
                self.condensing_set.append((point, label, 1))
            return

        for point, label in zip(new_data, new_labels):
            distances = [cdist([point], [p[0]], metric='euclidean') for p in self.condensing_set]
            nearest_idx = np.argmin(distances)
            nearest_proto = self.condensing_set[nearest_idx]
            if nearest_proto[1] == label:
                # Update weighted mean for matching prototype
                weight = nearest_proto[2]
                new_weighted_mean = (nearest_proto[0] * weight + point) / (weight + 1)
                self.condensing_set[nearest_idx] = (new_weighted_mean, label, weight + 1)
            else:
                # Add new prototype
                self.condensing_set.append((point, label, 1))

    def predict(self, data):
        """
        Predict labels for input data using a k-NN approach with KDTree for fast searching.
        """
        if not self.condensing_set:
            raise ValueError("Condensing set is empty. Train the model first.")
        
        tree = KDTree([proto[0] for proto in self.condensing_set])
        _, indices = tree.query(data)
        return np.array([self.condensing_set[idx][1] for idx in indices])

    def _recursive_cluster(self, data, labels):
        """
        Recursive k-means clustering to build a homogeneous condensing set.
        """
        queue = [(data, labels)]
        while queue:
            cluster_data, cluster_labels = queue.pop(0)
            unique_classes, class_counts = np.unique(cluster_labels, return_counts=True)
            majority_label = unique_classes[np.argmax(class_counts)]

            # Check homogeneity
            if np.max(class_counts) / len(cluster_labels) > self.homogeneity_threshold:
                centroid = np.mean(cluster_data, axis=0)
                self.condensing_set.append((centroid, majority_label, len(cluster_data)))
                continue

            # Determine number of clusters
            n_clusters = min(len(unique_classes), len(cluster_data))
            if self.max_clusters:
                n_clusters = min(n_clusters, self.max_clusters)

            # Prepare initial centers
            if len(unique_classes) >= n_clusters:
                initial_means = np.array([
                    np.mean(cluster_data[cluster_labels == cls], axis=0)
                    for cls in unique_classes[:n_clusters]
                ])
            else:
                # If unique_classes < n_clusters, use random points as initial centers
                random_indices = np.random.choice(len(cluster_data), n_clusters, replace=False)
                initial_means = cluster_data[random_indices]

            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, init=initial_means, n_init=1, random_state=42)
            cluster_assignments = kmeans.fit_predict(cluster_data)

            for cluster_idx in range(n_clusters):
                sub_data = cluster_data[cluster_assignments == cluster_idx]
                sub_labels = cluster_labels[cluster_assignments == cluster_idx]
                if len(sub_data) > 0:
                    queue.append((sub_data, sub_labels))


    def get_active_set_length(self):
        """
        Return the length of the condensing set.
        """
        return len(self.condensing_set)

    def fit_batch(self, data, labels):
        """
        Fit the model incrementally using batch data.
        """
        self.update_with_new_data(data, labels)


for i in range(1, 15):
    print(f"Starting dataset {i}")
    # Example usage
    data, labels, batch_size = list_and_select_dat_files('Datasets', i)
    print(f"Loaded dataset {i} with {len(data)} samples.")
    
    model = DRHC(max_clusters=15)
    print("Starting AIB fitting...")
    model.fit(data, labels)
    print("Finished AIB fitting.")

    print("Starting test-then-train evaluation...")
    results, cpu_time_table = test_then_train(
        model, data, labels, batch_size=batch_size)
    print("Finished evaluation.")
    
    for entry in results:
        print(entry)

    save_results_to_excel(results, "LWF", cpu_time_metrics=cpu_time_table)