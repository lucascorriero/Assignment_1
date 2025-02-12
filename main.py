from data import load_and_preprocess_data
from model_training import train_knn, train_decision_tree, train_random_forest
from plotting import plot_confusion_matrix

# Load and preprocess data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

# Train models and get results
results_knn = train_knn(X_train, X_test, y_train, y_test, scaler)
results_dt = train_decision_tree(X_train, X_test, y_train, y_test)
results_rf = train_random_forest(X_train, X_test, y_train, y_test)

# Print results
print("KNN Results: Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(*results_knn[:4]))
print("Decision Tree Results: Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(*results_dt[:4]))
print("Random Forest Results: Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(*results_rf[:4]))

# Plot confusion matrices
plot_confusion_matrix(results_knn[4], "KNN")
plot_confusion_matrix(results_dt[4], "Decision Tree")
plot_confusion_matrix(results_rf[4], "Random Forest")