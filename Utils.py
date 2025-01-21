import numpy as np
from openpyxl import Workbook
import pandas as pd
import os
import time
from sklearn.metrics import accuracy_score
import time


def test_then_train(model, data, labels, batch_size=50):
    """
    Function to process data in batches, test before training, track performance metrics, and measure CPU time.

    Args:
    - model: The instance of the IB1, IB2, or IB3 algorithm.
    - data: The dataset (numpy array or list of feature vectors).
    - labels: The labels corresponding to the dataset.
    - batch_size: Number of samples in each batch.

    Returns:
    - metrics_table: A list of dictionaries containing accuracy, reduction rate, and other metrics for each batch.
    - cpu_time_table: A list of dictionaries containing CPU time for each batch.
    """
    metrics_table = []
    cpu_time_table = []
    num_instances = len(data)

    # Split the data into batches
    for start_idx in range(0, num_instances, batch_size):
        end_idx = min(start_idx + batch_size, num_instances)
        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        # Record start time for CPU measurement
        start_time = time.process_time()

        # Test before training on the current batch
        if model.concept_description is not None and len(model.concept_description) > 0:
            predictions = model.predict(batch_data)
            accuracy = accuracy_score(batch_labels, predictions)
        else:
            accuracy = 0.0  # No prediction possible if model has not been trained yet

        # Train the model on the current batch
        model.fit(batch_data, batch_labels)
        
        # Record end time for CPU measurement
        end_time = time.process_time()
        cpu_time = end_time - start_time

        # Compute reduction rate (|CD| / |data| so far)
        reduction_rate = 1 - (len(model.concept_description)
                              if model.concept_description is not None else 0) / num_instances

        # Save metrics for this batch
        metrics_table.append({
            "Batch Start": start_idx,
            "Batch End": end_idx,
            "Accuracy": accuracy,
            "Reduction Rate": reduction_rate,
            "Concept Description Length": (len(model.concept_description) if model.concept_description is not None else 0)
        })

        # Save CPU time for this batch
        cpu_time_table.append({
            "Batch Start": start_idx,
            "Batch End": end_idx,
            "CPU Time (s)": cpu_time
        })

    return metrics_table, cpu_time_table


def combine_datasets(results_folder, output_file):
    """
    Reads all Excel files in the given folder and merges them into a single Excel file.
    Each dataset (Excel file) is separated by a blank row in the final file, and a 'Dataset' column
    is added to identify which dataset the rows came from.

    Parameters:
    - results_folder: Folder where the individual dataset Excel files are located.
    - output_file: Path to the final combined Excel file.
    """
    all_data = []

    # Loop over all Excel files in the directory
    for file in os.listdir(results_folder):
        if file.endswith(".xlsx"):
            file_path = os.path.join(results_folder, file)
            # Read the Excel file into a DataFrame
            df = pd.read_excel(file_path)

            # Add a 'Dataset' column to identify which dataset the rows came from (remove extension)
            dataset_name = os.path.splitext(file)[0]
            df['Dataset'] = dataset_name

            # Append a blank row DataFrame to separate this dataset from the next
            # We'll do this by appending df and then a single blank row df.
            all_data.append(df)
            # Create a single blank row using headers, but all values as None
            blank_row = pd.DataFrame({col: [None] for col in df.columns})
            all_data.append(blank_row)

    # Concatenate all data
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Remove trailing blank row if it exists
        if final_df.iloc[-1].isnull().all():
            final_df = final_df.iloc[:-1]

        # Write to Excel
        final_df.to_excel(output_file, index=False)
        print(f"Combined dataset saved to {output_file}")
    else:
        print("No Excel files found in directory.")


# Example usage:
# combine_datasets("./Results/DRHC", "./Results/DRHC/combined_datasets.xlsx")

def print_average_accuracy(results_folder):
    """
    Reads all Excel files in the given folder, calculates the average accuracy from each file,
    and prints the results.

    Parameters:
    - results_folder: Folder where the individual dataset Excel files are located.
    """
    # Loop over all Excel files in the directory
    for file in os.listdir(results_folder):
        if file.endswith(".xlsx"):
            file_path = os.path.join(results_folder, file)

            # Read the Excel file into a DataFrame
            df = pd.read_excel(file_path)

            # Ensure the 'Accuracy' column exists before calculating
            if 'Accuracy' in df.columns:
                avg_accuracy = df['Accuracy'].mean()
                dataset_name = os.path.splitext(file)[0]
                print(
                    f"Dataset: {dataset_name}, Average Accuracy: {avg_accuracy:.4f}")
            else:
                print(
                    f"Dataset: {file} does not contain an 'Accuracy' column.")

# Example usage:
# print_average_accuracy("./Results/DRHC")


def save_results_to_excel(results, algorithm, cpu_time_metrics=None):
    """
    Save results to an Excel file in a subfolder of ./Results named after the algorithm.
    Dynamically handles different keys in the results.
    If a file with the same name exists, increment the filename by 1.

    If CPU time metrics are provided, save them to a separate Excel file in the same folder.

    Parameters:
    - results: A list of dictionaries containing results.
    - algorithm: The algorithm name to use for the folder and filename.
    - cpu_time_metrics: Optional list of dictionaries containing CPU time metrics.
    """
    # Ensure the ./Results folder exists
    results_folder = "./Results"
    os.makedirs(results_folder, exist_ok=True)

    # Create a folder under ./Results for this specific algorithm
    alg_folder = os.path.join(results_folder, algorithm)
    os.makedirs(alg_folder, exist_ok=True)

    # Construct the initial filename inside the algorithm's folder for results
    results_filename = os.path.join(alg_folder, f"{algorithm}_results.xlsx")
    base_name, ext = os.path.splitext(results_filename)
    counter = 1
    while os.path.exists(results_filename):
        results_filename = f"{base_name}_{counter}{ext}"
        counter += 1

    # Save results to an Excel file
    if results:
        wb = Workbook()
        ws = wb.active
        ws.title = "Results"

        # Dynamically determine headers from the keys of the first result
        headers = list(results[0].keys())
        ws.append(headers)

        # Write data rows
        for result in results:
            ws.append([result.get(key, "") for key in headers])

        # Save the workbook
        wb.save(results_filename)
        print(f"Results saved to {results_filename}")

    # Save CPU time metrics if provided
    if cpu_time_metrics:
        cpu_time_filename = os.path.join(
            alg_folder, f"{algorithm}_cpu_time.xlsx")
        cpu_base_name, cpu_ext = os.path.splitext(cpu_time_filename)
        cpu_counter = 1
        while os.path.exists(cpu_time_filename):
            cpu_time_filename = f"{cpu_base_name}_{cpu_counter}{cpu_ext}"
            cpu_counter += 1

        # Save CPU time metrics to an Excel file
        wb_cpu = Workbook()
        ws_cpu = wb_cpu.active
        ws_cpu.title = "CPU Time"

        # Dynamically determine headers from the keys of the first CPU time metric
        cpu_headers = list(cpu_time_metrics[0].keys())
        ws_cpu.append(cpu_headers)

        # Write data rows
        for metric in cpu_time_metrics:
            ws_cpu.append([metric.get(key, "") for key in cpu_headers])

        # Save the workbook
        wb_cpu.save(cpu_time_filename)
        print(f"CPU Time metrics saved to {cpu_time_filename}")


def test_then_train(model, data, labels, batch_size=100):
    metrics_table = []
    cpu_time_table = []
    num_instances = len(data)

    for start_idx in range(0, num_instances, batch_size):
        end_idx = min(start_idx + batch_size, num_instances)
        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        start_time = time.perf_counter()

        # Ensure predictions are compatible with batch_labels
        if hasattr(model, "predict") and model.get_active_set_length() > 0:
            predictions = model.predict(batch_data)
            accuracy = accuracy_score(
                np.array(batch_labels).flatten().astype(int),
                np.array(predictions).flatten().astype(int)
            )
        else:
            accuracy = 0.0

        if hasattr(model, "fit"):
            model.fit(batch_data, batch_labels)
        else:
            raise AttributeError("The model does not have a 'fit' method.")

        reduction_rate = 1 - (model.get_active_set_length() / num_instances)
        end_time = time.perf_counter()
        cpu_time_ms = (end_time - start_time) * 1000

        metrics_table.append({
            "Batch Start": start_idx,
            "Batch End": end_idx,
            "Accuracy": accuracy,
            "Reduction Rate": reduction_rate,
            "Active Set Length": model.get_active_set_length()
        })

        cpu_time_table.append({
            "Batch Start": start_idx,
            "Batch End": end_idx,
            "CPU Time (ms)": cpu_time_ms
        })

        # if start_idx % 1000 == 0:
        print(f"Processed up to point {end_idx}.")
    return metrics_table, cpu_time_table
