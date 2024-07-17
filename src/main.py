# src/main.py
from data_collection import collect_from_database
from data_preprocessing import clean_data, transform_data
from model_training import train_model
from model_evaluation import eval_model
import joblib
import argparse
import pickle

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--retrain', action='store_true', help="Retrain the model if this flag is set.")
	args = parser.parse_args()

	# Collect data
	query = "SELECT * FROM CLAIMS.DS_DATASET"
	dataset_from_database = collect_from_database(query)

	# Clean and transform data
	dataset_from_database = clean_data(dataset_from_database)
	dataset_from_database = transform_data(dataset_from_database)
	X, y = dataset_from_database.drop('claim_status', axis=1), dataset_from_database[['claim_status']]

	# Train model
	model, X_train, X_val, y_train, y_val = train_model(args.retrain, X, y)

	if args.retrain:
		file_name = "xgboost_model_updated.pkl"
		pickle.dump(model, open(file_name, "wb"))

	# Evaluate model
	metrics = eval_model(model, X_train, X_val, y_train, y_val)

if __name__ == "__main__":
    main()
