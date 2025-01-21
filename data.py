import pandas as pd
import numpy as np
import os

import numpy as np

import numpy as np

def get_data(dataset):
    def encode_labels_to_numeric(labels):
        """
        Parameters:
            labels (numpy array): Array of alphanumeric labels.
        Returns:
            numpy array: Encoded numeric labels.
            dict: Mapping from original labels to numeric values (if applicable).
        """
        # Check if all labels can be converted to integers
        try:
            # Attempt to convert labels to integers
            numeric_labels = labels.astype(int)
            # If conversion succeeds, return the integer labels directly
            return numeric_labels, None
        except ValueError:
            # If conversion fails, encode alphanumeric labels
            unique_labels = np.unique(labels)
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = np.array([label_mapping[label] for label in labels])
            return numeric_labels, label_mapping

    path = f'Datasets/{dataset}'
    data = []
    labels = []

    # Parse the file
    with open(path, 'r') as file:
        is_data_section = False  # Flag to track if we are in the data section
        for line in file:
            line = line.strip()
            # Skip empty lines or comments
            if not line or line.startswith("@"):
                # Mark the start of the data section
                if line.lower() == "@data":
                    is_data_section = True
                continue

            # Process lines in the data section
            if is_data_section:
                parts = line.split(",")
                # Append features to data and the last value to labels
                data.append(list(map(float, parts[:-1])))  # Convert features to float
                labels.append(parts[-1])  # Last column is the label

    # Convert lists to numpy arrays
    data_array = np.array(data)
    labels_array = np.array(labels)

    # Attempt to convert labels to integers or encode them
    labels_array, label_mapping = encode_labels_to_numeric(labels_array)

    # Print label mapping if encoding was applied
    if label_mapping:
        print(f"Labels were alphanumeric and have been encoded with the following convention: {label_mapping}")

    return data_array, labels_array



def list_and_select_dat_files(directory, i=None):
    """
    Lists all .dat files in a directory and maps them to a user input or a given index.
    
    Parameters:
        directory (str): Path to the directory containing .dat files.
        i (int or None): Index of the dataset to select directly. If None, prompts user for input.
    
    Returns:
        tuple: The data, labels, and batch size from the selected dataset.
    """
    # Predefined batch sizes from the paper's buffer table
    batch_sizes = {
        "balance": 100,  # BL
        "banana": 530,   # BN
        "ecoli": 200,    # ECL
        "kddcup_numeric_fixed": 1000,  # KDD
        "letter": 2000,  # LR
        "magic": 1902,   # MGT
        "monk-2": 115,   # MN2
        "penbased": 1000,  # PD
        "phoneme": 500,  # PH
        "satimage": 572,  # LS
        "shuttle": 1856,  # SH
        "texture": 440,  # TXR
        "twonorm": 592,  # TN
        "yeast": 396,  # YS
    }

    # List all .dat files in the directory
    dat_files = [file for file in os.listdir(directory) if file.endswith(".dat")]

    # Check if no .dat files are found
    if not dat_files:
        print("No .dat files found in the specified directory.")
        return None, None, None

    # Display options to the user
    print("Available datasets:")
    for idx, file in enumerate(dat_files, start=1):
        print(f"{idx}. {file}")

    # Use the provided index or prompt for input
    if i is not None:
        if 1 <= i <= len(dat_files):
            selected_file = dat_files[i - 1]
        else:
            print(f"Invalid index. Please provide a number between 1 and {len(dat_files)}.")
            return None, None, None
    else:
        while True:
            try:
                choice = int(input("Select a dataset by entering the corresponding number: "))
                if 1 <= choice <= len(dat_files):
                    selected_file = dat_files[choice - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(dat_files)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    # Extract dataset identifier (e.g., 'balance', 'banana') from the filename
    dataset_key = selected_file.split(".")[0].lower()  # Lowercase for consistency
    
    # Fetch the batch size for the dataset
    batch_size = batch_sizes.get(dataset_key, None)

    if batch_size is None:
        print(f"Batch size for dataset {dataset_key} is not defined.")
        return None, None, None

    # Call get_data with the selected file
    data, labels = get_data(selected_file)
    return data, labels, batch_size


# Example usage
# list_and_select_dat_files('Datasets')
