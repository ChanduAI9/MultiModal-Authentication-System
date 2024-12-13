from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def tune_knn(X_train, y_train):
    """
    Perform hyperparameter tuning for kNN.

    Parameters:
    - X_train: Training features.
    - y_train: Training labels.

    Returns:
    - Best kNN model.
    """
    param_grid = {"n_neighbors": [3, 5, 7], "metric": ["euclidean", "manhattan"]}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(f"Best Parameters for kNN: {grid.best_params_}")
    return grid.best_estimator_

def tune_svm(X_train, y_train):
    """
    Perform hyperparameter tuning for SVM.

    Parameters:
    - X_train: Training features.
    - y_train: Training labels.

    Returns:
    - Best SVM model.
    """
    param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    grid = GridSearchCV(SVC(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(f"Best Parameters for SVM: {grid.best_params_}")
    return grid.best_estimator_
