import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from Utils import save_results_to_excel, test_then_train
from data import *
import numpy as np
from sklearn.metrics import accuracy_score


class IB1:
    def __init__(self):
        self.concept_description = None
        self.labels = None

    def fit(self, data, labels):
        if self.concept_description is not None:
            self.concept_description = np.vstack(
                (self.concept_description, np.array(data)))
            self.labels = np.hstack((self.labels, np.array(labels)))
        else:
            self.concept_description = np.array(data)
            self.labels = np.array(labels)

    def predict(self, data):
        indices, _ = pairwise_distances_argmin_min(
            data, self.concept_description)
        return self.labels[indices]

    def get_active_set_length(self):
        return len(self.concept_description) if self.concept_description is not None else 0


class IB2:
    def __init__(self):
        self.concept_description = None
        self.labels = None

    def fit(self, data, labels):
        data = np.array(data)
        labels = np.array(labels)

        if self.concept_description is None:
            self.concept_description = np.array([data[0]])
            self.labels = np.array([labels[0]])

        for i in range(1, len(data)):
            sample = data[i]
            label = labels[i]

            nearest_idx, _ = pairwise_distances_argmin_min(
                sample.reshape(1, -1), self.concept_description)
            nearest_label = self.labels[nearest_idx[0]]

            if nearest_label != label:
                self.concept_description = np.vstack(
                    (self.concept_description, sample))
                self.labels = np.hstack((self.labels, label))

    def predict(self, data):
        indices, _ = pairwise_distances_argmin_min(
            data, self.concept_description)
        return self.labels[indices]

    def get_active_set_length(self):
        return len(self.concept_description) if self.concept_description is not None else 0


class IB3:
    def __init__(self, accept_conf=0.90, discard_conf=0.75):
        self.concept_description = []
        self.labels = []
        self.performance_records = []
        self.accept_conf = accept_conf
        self.discard_conf = discard_conf

    def _confidence(self, successes, trials):
        if trials == 0:
            return 0, 1
        p_hat = successes / trials
        z = 1.96
        error_margin = z * np.sqrt((p_hat * (1 - p_hat)) / trials)
        return max(0, p_hat - error_margin), min(1, p_hat + error_margin)

    def fit(self, data, labels):
        for i in range(len(data)):
            instance = data[i].reshape(1, -1)
            label = labels[i]

            if len(self.concept_description) > 0:
                indices, _ = pairwise_distances_argmin_min(
                    instance, np.array(self.concept_description))
                nearest_idx = indices[0]
                nearest_label = self.labels[nearest_idx]

                if nearest_label == label:
                    self.performance_records[nearest_idx][0] += 1
                else:
                    self.performance_records[nearest_idx][1] += 1

                correct, incorrect = self.performance_records[nearest_idx]
                total = correct + incorrect
                lower, upper = self._confidence(correct, total)
                class_prob = np.mean(np.array(self.labels) == nearest_label)

                if upper < class_prob:
                    del self.concept_description[nearest_idx]
                    del self.labels[nearest_idx]
                    del self.performance_records[nearest_idx]
                    continue

            if len(self.concept_description) == 0 or nearest_label != label:
                self.concept_description.append(instance.flatten())
                self.labels.append(label)
                self.performance_records.append([1, 0])

    def predict(self, data):
        indices, _ = pairwise_distances_argmin_min(
            data, np.array(self.concept_description))
        return np.array([self.labels[idx] for idx in indices])

    def get_active_set_length(self):
        return len(self.concept_description)


for i in range(1, 15):
    print(f"Starting dataset {i}")
    # Example usage
    data, labels, batch_size = list_and_select_dat_files('Datasets', i)
    print(f"Loaded dataset {i} with {len(data)} samples.")
    
    model = IB3()
    print("Starting AIB fitting...")
    model.fit(data, labels)
    print("Finished AIB fitting.")

    print("Starting test-then-train evaluation...")
    results, cpu_time_table = test_then_train(
        model, data, labels, batch_size=batch_size)
    print("Finished evaluation.")
    
    for entry in results:
        print(entry)

    save_results_to_excel(results, "IB3", cpu_time_metrics=cpu_time_table)
