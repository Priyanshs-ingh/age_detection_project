Health Prediction in Urban Areas
Project Description
This project uses machine learning to predict health risks associated with smoking in urban areas. By considering factors like age, location, and smoking habits, it provides personalized health assessments and insights into potential health risks.    

Key Features
Predicts health risks based on individual smoking habits.
Provides personalized health risk assessments.
Raises awareness about the effects of smoking.
Serves as a preventive tool to motivate informed decisions about smoking.    
Technologies Used
Python
Pandas
Scikit-learn (RandomForestRegressor, StandardScaler, OneHotEncoder, GridSearchCV)
SHAP (for feature importance analysis)    
Methodology
Data Import and Processing:

Import data from Our World in Data using Pandas.
Create a 'Health_Index' by combining disease and smoking data from 1990 and 2021.    
Model Selection and Setup:

Use RandomForestRegressor from scikit-learn.
Split data into training (80%) and testing (20%) sets.    
Data Preprocessing:

Normalize numerical data using StandardScaler.
Encode categorical variables using OneHotEncoder.
Implement preprocessing pipelines.    
Model Training and Evaluation:

Train the model and evaluate performance using Mean Squared Error (MSE).    
Parameter Optimization:

Conduct hyperparameter tuning using GridSearchCV.    
Feature Importance Analysis:

Analyze influential predictors using SHAP values and permutation tests.
Employ cross-validation.    
Final Model Assessment:

Calculate final MSE on the test set and interpret results.    


Dataset
The dataset used for this project is sourced from Our World in Data and focuses on smoking-related death rates across different countries and regions over time.    

Future Enhancements
Incorporate additional risk factors (e.g., diet, exercise, air quality).
Develop a user-friendly web interface for easier access.
Expand the dataset to include more diverse populations.
Acknowledgments
Our World in Data for providing the valuable dataset.
Scikit-learn for the machine learning tools.
