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

## Additional Questions for the Task

### What are the assumptions you have made for this service and why?

- I have assumed the data querying process requires no change, just no real time to change the process from it's original query design.

### What considerations are there to ensure the business can leverage this service?

- So the code is containerisable but my limited experience with cloud services means I'd need to see further how it would integrate into one from a data perspective.
- The unit tests prove that to have a fully robust code in a production environment, a lot more work would be required on the python functions.
- I need more information about the data and actual process so we can build the timing of pipelines more accurately.

### Which traditional teams within the business would you need to talk to and why?

- The data team, it looks like we have everything we need from the description but it's so key, ensuring what you think is true is definitely worth doing. 
- The IT team, if not known already, see what IT infrastructure is available and/or strategy is working towards to ensure the system we build is compatible with the overriding AI and data strategy.
- Head of claims, get as much information of Trevor as possible to ensure in fact what he says he wants, it is indeed and we can't offer something better.

### What is in and out of scope for your responsibility?

- Building a high performance generalisable model.
- Model build documentation for any governance process.
- Performing explatory data analysis of the data.

