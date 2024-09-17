from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Example: Train a model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, r'C:\Users\priyansh singh\Documents\GitHub\age_detection_project\model.pkl')

# Later...load the model
loaded_model = joblib.load(r'C:\Users\priyansh singh\Documents\GitHub\age_detection_project\model.pkl')
