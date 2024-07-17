# src/model_training.py
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils import class_weight
import xgboost as xgb
from scipy import stats
import json
from sklearn.preprocessing import LabelEncoder
import pickle

def train_model(retrain, X, y):
	# Split data
	le = LabelEncoder()
	X_train, X_val, y_train, y_val = train_test_split(X, le.fit_transform(y.values.ravel()), test_size=0.2, random_state=42)

	if retrain:
		print("Retraining the model...")

		# Eval stuff
		eval_metrics = ["auc", "rmse", "logloss"]
		eval_set = [(X_val, y_val)]

		# Class weighting
		classes_weights = class_weight.compute_sample_weight(
		    class_weight="balanced",
		    y=y_train
		)

		# Get optimal hyperparameter
		print("Finding hyperparamaters")
		parameter_gridSearch = RandomizedSearchCV(
		    estimator=xgb.XGBClassifier(
		    objective='binary:logistic',
		    eval_metric=eval_metrics,
		    early_stopping_rounds=15,
		    enable_categorical=True
		    ),

		    param_distributions={
		    'n_estimators': stats.randint(50, 500),
		    'learning_rate': stats.uniform(0.01, 0.75),
		    'subsample': stats.uniform(0.25, 0.75),
		    'max_depth': stats.randint(1, 8),
		    'colsample_bytree': stats.uniform(0.1, 0.75),
		    'min_child_weight': [1, 3, 5, 7, 9],
		    },

		    cv=5,
		    n_iter=100,
		    verbose=False,
		    scoring='roc_auc',
		)

		# Train the gridsearch
		parameter_gridSearch.fit(X_train, y_train, eval_set=eval_set, sample_weight=classes_weights, verbose=False)

		# Train final model
		print("Fitting final model")
		model = xgb.XGBClassifier(
		    objective='binary:logistic',
		    eval_metric=eval_metrics,
		    early_stopping_rounds=15,
		    enable_categorical=True,
		    **parameter_gridSearch.best_params_
		    )

		# Train final model
		model.fit(X_train, y_train, eval_set=eval_set, sample_weight=classes_weights, verbose=False)
	else:
		print("Using existing model...")

		# load
		file_name = "xgboost_model_updated.pkl"
		model = pickle.load(open(file_name, "rb"))

	return model, X_train, X_val, y_train, y_val
