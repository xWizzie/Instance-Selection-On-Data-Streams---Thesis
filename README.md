# Instance Selection on Data Streams - Thesis

This repository contains the code and resources developed for the thesis on "Instance Selection on Data Streams." The work focuses on implementing and evaluating various instance selection algorithms tailored for data streams, addressing challenges such as concept drift and the need for real-time processing.

## Overview

Instance selection is a crucial preprocessing step in machine learning, especially when dealing with large-scale data streams. It aims to reduce the dataset size by selecting a representative subset of instances, thereby improving computational efficiency and potentially enhancing model performance. This thesis explores different instance selection methods and their applicability to dynamic data streams.

## Repository Structure

- **`data.py`**: Handles data loading, preprocessing, and management for streaming scenarios.
- **`Utils.py`**: Contains utility functions to support various operations within the project.
- **`aib2.py`**, **`drhc4.py`**, **`ib.py`**, **`lwf.py`**, **`pecs.py`**, **`twf.py`**: Implementations of different instance selection algorithms evaluated in the thesis.

## Instance Selection Algorithms

The repository includes implementations of several instance selection algorithms:

- **IB1-3**: A standard instance-based learning algorithm adapted for streaming data.
- **TWF**: The Time-Weighted Function classifier, emphasizing recent data in streams.
- **LWF**: The Locally-Weighted Forgetting algorithm, which handles evolving data distributions.
- **PECS**: Prototype Evolution for Classification in data Streams, focusing on maintaining representative prototypes.
- **DRHC**: A dynamic recursive hybrid classifier tailored for concept drift scenarios.
- **AIB**: An advanced instance-based algorithm designed for efficient data stream processing.

These algorithms are implemented with considerations for data streams, ensuring they can handle the challenges of streaming data, such as concept drift and limited memory.

## Getting Started

To run the code in this repository, ensure you have Python installed. Clone the repository and install any necessary dependencies:

```bash
git clone https://github.com/xWizzie/Instance-Selection-On-Data-Streams---Thesis.git
cd Instance-Selection-On-Data-Streams---Thesis
pip install -r requirements.txt
```

## Usage

Each algorithm implementation can be executed independently. For example, to run the `aib2.py` script:

```bash
python aib2.py
```

Ensure that the data required for the experiments is properly formatted and located in the appropriate directory as specified in `data.py`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
