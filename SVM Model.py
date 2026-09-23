import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Generate a sample dataset (replace X_train, y_train with your real data)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features (Critical for SVM performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define the robust hyperparameter grid to find the "best model" configuration
param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1, 1]}
]

# 5. Initialize Grid Search with 5-fold Cross-Validation
base_svc = SVC(random_state=42)
grid_search = GridSearchCV(estimator=base_svc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# 6. Train the model to find optimal parameters
grid_search.fit(X_train_scaled, y_train)

# 7. Extract the best model
best_svm_model = grid_search.best_estimator_
print("Best Parameters Found:", grid_search.best_params_)

# 8. Evaluate on test data
y_pred = best_svm_model.predict(X_test_scaled)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
