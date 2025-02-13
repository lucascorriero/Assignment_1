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

### how to get default results

- copy this into model_training.py 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_knn(X_train, X_test, y_train, y_test, scaler):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return evaluate_model(y_test, y_pred)

def train_decision_tree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return evaluate_model(y_test, y_pred)

def train_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_traian, y_train)
    y_pred = rf.predict(X_test)
    return evaluate_model(y_test, y_pred)

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, cm


## How to Run

1. Clone the repository:
   ```sh
   git clone https://github.com/lucascorriero/Assignment_1
   cd Assignment_1
2 Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
3 Install the dependencies:
   pip install scikit-learn matplotlib seaborn
4 Run the main script:
   python main.py
