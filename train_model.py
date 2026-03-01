import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json
import warnings
warnings.filterwarnings('ignore')


print("Loading dataset...")
diabetes_dataset = pd.read_csv("diabetes.csv")

# Display first five rows
print("\nFirst 5 rows:")
print(diabetes_dataset.head())

# Check class distribution
print("\nClass distribution:")
print(diabetes_dataset["Outcome"].value_counts())

# Check mean values by outcome
print("\nMean values by Outcome:")
print(diabetes_dataset.groupby("Outcome").mean())


print("\nHandling zero values (missing data)...")
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in zero_columns:
    # Replace zeros with NaN FIRST
    diabetes_dataset[col] = diabetes_dataset[col].replace(0, np.nan)
    # Impute using median GROUPED by Outcome (preserves class distribution)
    diabetes_dataset[col] = diabetes_dataset.groupby('Outcome')[col].transform(
        lambda x: x.fillna(x.median())
    )

# Fill any remaining NaNs
diabetes_dataset.fillna(diabetes_dataset.median(), inplace=True)


print("\nCreating new features...")
diabetes_dataset['Glucose_Age'] = diabetes_dataset['Glucose'] * diabetes_dataset['Age'] / 100
diabetes_dataset['BMI_Insulin'] = diabetes_dataset['BMI'] * diabetes_dataset['Insulin'] / 100

# =============================================================================
# PREPARE DATA
# =============================================================================
X = diabetes_dataset.drop(columns='Outcome')
y = diabetes_dataset['Outcome']

print(f"\nFeature shape: {X.shape}")
print(f"Target shape: {y.shape}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")


print("\nCreating pipeline...")
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
])


print("\nStarting hyperparameter tuning...")
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 0.001, 0.01, 0.1, 1],
    'svm__class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)


print("\nTraining optimized SVM...")
grid_search.fit(X_train, y_train)


print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Best Parameters: {grid_search.best_params_}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

# Save the BEST ESTIMATOR (not grid_search object)
joblib.dump(best_model, 'diabetes_model.pkl')
print(" Model saved as 'diabetes_model.pkl'")

# Save feature names for reference
feature_names = X.columns.tolist()
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print(" Feature names saved as 'feature_names.json'")

# Save grid search results for reference
grid_results = {
    'best_params': grid_search.best_params_,
    'best_score': float(grid_search.best_score_),
    'cv_results': {
        'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
        'std_test_score': grid_search.cv_results_['std_test_score'].tolist()
    }
}
with open('grid_search_results.json', 'w') as f:
    json.dump(grid_results, f, indent=2)
print(" Grid search results saved as 'grid_search_results.json'")

print("\nModel saved successfully!")


print("\n" + "="*60)
print("VERIFICATION - LOADING AND TESTING SAVED MODEL")
print("="*60)

# Load the saved model
loaded_model = joblib.load('diabetes_model.pkl')

# Verify it's a pipeline object (not numpy array)
print(f"\nLoaded model type: {type(loaded_model)}")
print(f"Has predict method: {hasattr(loaded_model, 'predict')}")

# Test prediction with loaded model
y_pred_loaded = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)

print(f"\nLoaded Model Test Accuracy: {loaded_accuracy:.4f} ({loaded_accuracy*100:.2f}%)")
print(f"Accuracy matches original: {loaded_accuracy == test_accuracy}")


def predict_diabetes(model_path, feature_names_path, patient_data):
    """
    Load model and make prediction for new patient data.
    
    Parameters:
    -----------
    model_path : str - Path to saved model
    feature_names_path : str - Path to feature names JSON
    patient_data : dict - Patient feature values
    
    Returns:
    --------
    dict - Prediction result with probability
    """
    # Load model
    model = joblib.load(model_path)
    
    # Load feature names
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    # Create DataFrame from patient data
    patient_df = pd.DataFrame([patient_data], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0][1]
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'diagnosis': 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    }

# Example usage (uncomment to test)
# sample_patient = {
#     'Pregnancies': 1, 'Glucose': 85, 'BloodPressure': 66, 'SkinThickness': 29,
#     'Insulin': 0, 'BMI': 26.6, 'DiabetesPedigreeFunction': 0.351, 'Age': 31,
#     'Glucose_Age': 26.35, 'BMI_Insulin': 0.0
# }
# result = predict_diabetes('diabetes_model.pkl', 'feature_names.json', sample_patient)
# print(f"\nSample Prediction: {result}")

print("\n" + "="*60)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*60)