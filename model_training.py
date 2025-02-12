from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_knn(X_train, X_test, y_train, y_test, scaler):
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return evaluate_model(y_test, y_pred)

def train_decision_tree(X_train, X_test, y_train, y_test, max_depth=None):
    dt = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return evaluate_model(y_test, y_pred)

def train_random_forest(X_train, X_test, y_train, y_test, max_depth=None):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=max_depth)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return evaluate_model(y_test, y_pred)

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, cm