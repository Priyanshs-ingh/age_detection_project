import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Assuming a regression model
from sklearn.metrics import mean_squared_error
import numpy as np
data = pd.read_csv('merged_data.csv')
# Verify and print all column names to confirm the disease and smoking columns
print(data.columns)
disease_columns = [col for col in data.columns if col.startswith('Disease')]
data['Health_Index'] = (data[disease_columns].sum(axis=1) + (data['Smoking_2021'] + data['Smoking_1990']) / 2) / (len(disease_columns) + 1)
y = data['Health_Index']  # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
from sklearn.preprocessing import StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
predictions_scaled = model.predict(X_test_scaled)
mse_scaled = mean_squared_error(y_test, predictions_scaled)
print(f'Scaled Mean Squared Error: {mse_scaled}')
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best MSE from Grid Search: {-grid_search.best_score_}')
import pandas as pd
import matplotlib.pyplot as plt
residuals = y_test - predictions_scaled
plt.scatter(predictions_scaled, residuals)
print(data.columns)  # This will list all the column names in your DataFrame
print(high_leverage_points)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, RandomForestRegressor(random_state=42, n_estimators=300, max_depth=10))
pipeline.fit(X_train, y_train)
predictions_robust = pipeline.predict(X_test)
mse_robust = mean_squared_error(y_test, predictions_robust)
print(f'Robust Scaled Mean Squared Error: {mse_robust}')
print("Feature Statistics:")
print(data[disease_columns + ['Smoking_1990', 'Smoking_2021']].describe())
print("\nNigeria Specific Values:")
print(data.loc[56, disease_columns + ['Smoking_1990', 'Smoking_2021']])
import numpy as np
data['Health_Index'] = (data[disease_columns].mean(axis=1) + data['Average_Smoking']) / 2
y = data['Health_Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Re-fit and evaluate the model
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Adjusted Mean Squared Error: {mse}')
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
print(f'Average MSE from Cross-Validation: {average_mse}')
import shap
# Plot SHAP values for the first 10 predictions
import matplotlib.pyplot as plt
# Get feature importances
importances = model.feature_importances_
# Plotting feature importances
plt.barh(feature_names, importances, color='skyblue')
from sklearn.inspection import permutation_importance
# Perform permutation importance
results = permutation_importance(model, X_train, y_train, scoring='neg_mean_squared_error')
# Get importance scores
importance_scores = results.importances_mean
sorted_idx = np.argsort(importance_scores)
plt.barh(range(len(sorted_idx)), importance_scores[sorted_idx], align='center')
from sklearn.utils import resample
    model.fit(X_resample, y_resample)
print("Bootstrap Scores Mean:", np.mean(bootstrap_scores))
print("Bootstrap Scores Std Dev:", np.std(bootstrap_scores))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
data = pd.read_csv('merged_data.csv')  # This should include a 'Country' column
data['Health_Index'] = (data[disease_columns].mean(axis=1) + data['Average_Smoking']) / 2
if 'Country' not in data.columns:
X = data.drop('Health_Index', axis=1)
y = data['Health_Index']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
data = pd.read_csv('merged_data.csv')
print("Unique countries in dataset:", data['Country'].unique())
# OneHotEncoder setup for the 'Country' feature with handle_unknown parameter
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
# Define the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
pipeline = Pipeline(steps=[
data['Health_Index'] = (data[disease_columns].mean(axis=1) + data['Average_Smoking']) / 2
X = data.drop('Health_Index', axis=1)
y = data['Health_Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
pipeline.fit(X_train, y_train)
# Make predictions and calculate mean squared error
predictions = pipeline.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
preprocessor = ColumnTransformer(
        ('num', Pipeline([
        ('cat', OneHotEncoder(), ['Country'])
model = Pipeline([
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Mean Cross-Validated MSE: ", np.mean(np.abs(scores)))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Final Test Set Mean Squared Error: {mse}')
from joblib import dump, load
# # %%
# 
# # Load data
# 
# 
# # Identify all columns that are disease metrics (assuming they start with 'Disease')
# 
# # Calculate the Health Index
# weighting_factor_smoking = 1.5  # Smoking might be more impactful, this is arbitrary and for demonstration
# number_of_factors = len(disease_columns) + 1  # Number of disease metrics plus one for smoking
# 
# # Adjusting the Health Index to normalize the smoking impact
# 
# 
# # Prepare features and target variable
# X = data[disease_columns + ['Smoking_2021']]  # features include all disease columns and smoking
# 
# # Split the data into training and testing sets
# 
# # Initialize and train the model
# 
# # Predict and evaluate
# 
# 
# 
# # %%
# 
# # Standardizing the features
# scaler = StandardScaler()
# X_test_scaled = scaler.transform(X_test)
# 
# # Retrain the model with scaled data
# 
# 
# 
# # %%
# 
# # Setting up the parameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# 
# # Running a grid search to find the best parameters
# 
# 
# 
# # %%
# 
# plt.xlabel('Predictions')
# plt.ylabel('Residuals')
# plt.title('Residual Plot')
# plt.axhline(y=0, color='r', linestyle='--')
# plt.show()
# 
# 
# # %%
# 
# 
# # %%
# # Identifying potential outliers
# data['residuals'] = residuals  # Add residuals to the DataFrame
# high_leverage_points = data[np.abs(data['residuals']) > 15000]  # Adjust the threshold based on residual distribution
# 
# 
# # %%
# 
# # Scaling the features using RobustScaler
# scaler = RobustScaler()
# 
# # Create a pipeline with RobustScaler and RandomForest
# 
# # Fit the model with robust scaling
# 
# 
# # Check feature values for Nigeria and compare with dataset statistics
# 
# 
# # %%
# 
# # Apply a logarithmic transformation to the disease metrics and smoking data
# for col in disease_columns + ['Smoking_1990', 'Smoking_2021']:
#     data[col] = np.log1p(data[col])  # log1p is used to handle zero values gracefully
#     
# # Calculate the average smoking metric from both 1990 and 2021 data
# data['Average_Smoking'] = (data['Smoking_1990'] + data['Smoking_2021']) / 2
# 
# # Recalculate the Health Index after transformation
# 
# # Re-split the data
# X = data[disease_columns + ['Smoking_1990', 'Smoking_2021']]
# 
# 
# 
# 
# # %%
# 
# # Using cross-validation to evaluate the model
# average_mse = -scores.mean()
# 
# 
# 
# # %%
# 
# # Create the explainer - assumes the model and X_train from previous steps are still in memory
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_train)
# 
# shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=10)
# 
# 
# # %%
# 
# feature_names = X.columns
# 
# plt.figure(figsize=(10, 6))
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.title('Feature Importance from RandomForest')
# plt.show()
# 
# 
# # %%
# 
# 
# 
# # Visualize the results
# plt.yticks(range(len(sorted_idx)), [X_train.columns[i] for i in sorted_idx])
# plt.xlabel("Permutation Importance (mean decrease in MSE)")
# plt.show()
# 
# 
# # %%
# 
# bootstrap_scores = []
# 
# # Performing 100 bootstrap resamples
# for _ in range(100):
#     X_resample, y_resample = resample(X_train, y_train)
#     score = model.score(X_test, y_test)  # Or use any other appropriate performance metric
#     bootstrap_scores.append(score)
# 
# 
# 
# # %%
# 
# # Assume loading a dataset that needs country data added or it's already included
# 
# # Calculate the average smoking metric from both 1990 and 2021 data
# data['Average_Smoking'] = (data['Smoking_1990'] + data['Smoking_2021']) / 2
# # Recalculate the Health Index after transformation
# # Simulate 'Country' data if not present (for demonstration)
#     countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'India', 'China', 'Japan']
#     data['Country'] = np.random.choice(countries, size=len(data))
# 
# # Assuming age data is also needed
# np.random.seed(42)
# data['Age'] = np.random.randint(20, 80, size=len(data))
# data['Age_Impact'] = data['Age'].apply(lambda x: np.exp(-(x-50)**2/100))  # Example impact function
# 
# # Features and Target
# 
# # Preprocessing for categorical data
# categorical_features = ['Country']
# 
# # Combine categorical and numerical preprocessing
#     transformers=[
#         ('cat', categorical_transformer, categorical_features)],
#     remainder='passthrough')
# 
# # Create the RandomForest model pipeline
# 
# # Split the data into training and testing sets
# 
# # Fit the model
# 
# # Predict and evaluate the model
# 
# 
# # %%
# 
# # Load your dataset
# 
# # Optionally check unique countries if necessary to understand the data
# 
# # Prepare the data with the necessary transformations
# 
# # StandardScaler for 'Age' to handle numerical data
# scaler = StandardScaler()
# 
# # Define the Column Transformer for combined preprocessing
#     transformers=[
#         ('num', scaler, ['Age']),  # Scale the 'Age' column
#         ('cat', onehot_encoder, ['Country'])  # One-hot encode the 'Country' column
#     ],
#     remainder='passthrough'  # Include all other columns as they are
# )
# 
# 
# # Create a pipeline that first transforms the data and then applies the model
#     ('preprocessor', preprocessor),
#     ('regressor', model)
# ])
# # Assuming age data is also needed
# np.random.seed(42)
# data['Age'] = np.random.randint(20, 80, size=len(data))
# data['Age_Impact'] = data['Age'].apply(lambda x: np.exp(-(x-50)**2/100))  # Example impact function
# 
# # Calculate the average smoking metric from both 1990 and 2021 data
# data['Average_Smoking'] = (data['Smoking_1990'] + data['Smoking_2021']) / 2
# # Recalculate the Health Index after transformation
# # Simulate 'Country' data if not present (for demonstration)
# # Define features and target
# 
# # Split the data into training and testing sets
# 
# # Fit the model
# 
# 
# 
# # %%
# 
# # Data preprocessing
#     transformers=[
#             ('scaler', StandardScaler()),
#             ('poly', PolynomialFeatures(degree=2))
#         ]), ['Age']),
#     ])
# 
# # Model setup
#     ('preprocessor', preprocessor),
#     ('regressor', GradientBoostingRegressor(n_estimators=200, random_state=42))
# ])
# 
# # Train and evaluate using cross-validation
# 
# # Fit the model on full training data and evaluate on test set
# 
# 
# # %%
# 
# # Save the model to disk
# dump(model, 'health_index_model.joblib')
# 
# # To load the model back (e.g., during deployment)
# model = load('health_index_model.joblib')
# 
# 
# 
