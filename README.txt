Requirements
pip install numpy pandas scikit-learn matplotlib seaborn

# Machine Learning Model Training and Evaluation

This project demonstrates how to train and evaluate machine learning models using K-Nearest Neighbors (KNN), Decision Tree, and Random Forest classifiers. The project includes functions to train these models, evaluate their performance, and plot confusion matrices with additional metrics.

## Files

- `data.py`: Contains functions to load and preprocess the dataset.
- `model_training.py`: Contains functions to train KNN, Decision Tree, and Random Forest models, and evaluate their performance.
- `plotting.py`: Contains functions to plot confusion matrices with additional metrics.
- `main.py`: Main script to load data, train models, evaluate them, and plot the results.

## Functions

### model_training.py

- `train_knn(X_train, X_test, y_train, y_test, scaler)`: Trains a KNN model with `n_neighbors=7` and returns evaluation metrics.
- `train_decision_tree(X_train, X_test, y_train, y_test, max_depth=None)`: Trains a Decision Tree model with an optional `max_depth` parameter and returns evaluation metrics.
- `train_random_forest(X_train, X_test, y_train, y_test, max_depth=None)`: Trains a Random Forest model with an optional `max_depth` parameter and returns evaluation metrics.
- `evaluate_model(y_test, y_pred)`: Evaluates the model's performance and returns accuracy, precision, recall, F1-score, and confusion matrix.

### plotting.py

- `plot_confusion_matrix(y_true, y_pred, title, accuracy, precision, recall, f1, n_neighbors=None)`: Plots a confusion matrix with additional metrics.

## How to Run

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
