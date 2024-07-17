# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def clean_data(df):
	print(f"Preprocessing data")

	try:
		# Drop variables following missing
		df.drop(columns=['family_history_3', 'employment_type'], inplace=True)
	except:
		pass

	try:
		# Drop variables following correlation
		df.drop(columns=['financial_hist_1', 'insurance_hist_1', 'insurance_hist_2', 'bmi'], inplace=True)
	except:
		pass

	# Category conversion
	categorical_features = ["marital_status", "occupation", "location", 'gender', 'prev_claim_rejected', 'known_health_conditions', 'uk_residence', 'family_history_1', 'family_history_2', 'family_history_4', 'family_history_5', 'product_var_1', 'product_var_2', 'product_var_3', 'health_status', 'driving_record', 'previous_claim_rate', 'education_level', 'income level', 'n_dependents']

	for column in categorical_features:
		df[column] = df[column].astype('category')

	return df

def transform_data(df):
	# Categorical transformer
	categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

	# Identify categorical features
	categorical_features = df.select_dtypes(include=['category']).columns.tolist()

	# Identify numerical features
	numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

	# Numerical transformer
	numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

	# Combine preprocessing steps
	preprocessor = ColumnTransformer(
		transformers=[
			('cat', categorical_transformer, categorical_features),
			('num', numerical_transformer, numerical_features)
		],
		remainder='passthrough',  # Keep other columns unchanged
		verbose_feature_names_out=False  # Disable prefixing
	)

	# Apply transformations and convert to DataFrame
	transformed_data = preprocessor.fit_transform(df)
	transformed_df = pd.DataFrame(transformed_data, columns=preprocessor.get_feature_names_out())

	return transformed_df