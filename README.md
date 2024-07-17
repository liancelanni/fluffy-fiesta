# fluffy-fiesta

## Overview
This project aims to predict the likelihood of claims for insurance applications.

## Setup Instructions

### Prerequisites
- Python 3.6+
- Docker 

### Installation
1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Project
To run the main script, use:
```sh
python src/main.py
```

### Run Tests
To run the tests, use:
```sh
pytest tests/
```

### Using Docker

Build the Docker image:
```sh
docker build -t my-python-project .
```

Run the Docker container:
```sh
docker run -it my-python-project
```

## Project Details

### Source Code
- `src/main.py`: Entry point for the project.
- `src/data_collection.py`: Script to collect data.
- `src/data_preprocessing.py`: Script to clean and transform the data.
- `src/model_training.py`: Script to train the model.
- `src/model_evaluation.py`: Script to evaluate the model performance.

### Source Code
- `notebooks/ML_Engineer_Task.ipynb`: Jupyter notebook detailing the machine learning task.
- `notebooks/xgboost_model_optimised_with_cross_validation.json`: Original optimised XGBoost model with cross-validation.

### Tests
- `tests/test_data_processing.py`: Unit tests for data processing.
- `tests/test_model_training.py`: Unit tests for model training.



