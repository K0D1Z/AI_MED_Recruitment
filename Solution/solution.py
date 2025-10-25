# Import module to read data from csv file
import csv

# Import modules for preprocessing and splitting data
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Import Classical Machine Learning Methods
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Import Classification Quality Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score)

# Import Cross-Validation feature
from sklearn.model_selection import StratifiedKFold
# I use StratifiedKFold to ensure
# that the label data is always split in
# an equal proportion for each fold test
# as it is in the entire dataset

# Import pandas for table summary
import pandas as pd

# Constant variables
FILENAME = "task_data.csv"
TARGET_COLUMN = "Cardiomegaly"
RANDOM_STATE = 123  # To guarantee the same results for each program run
MAX_ITER = 1000  # Max interation in Logistic Regression
N_SPLITS = 5  # Number of CV tests (nr of folds)


# Some strings contain comma instead of dot
def convert_comma_separated_number(number):
    number = number.replace(",", ".")
    return float(number)


x_data_dicts = []  # List of samples in dictionaries
y_data = []  # List of labels

# Extract, preprocess and load data into separate lists of labels and samples

with open(FILENAME, "r") as csvfile:
    data = csv.DictReader(csvfile)
    for row in data:
        # Convert string with comma into float using convert_comma_separated_number() function

        row["CTR - Cardiothoracic Ratio"] = convert_comma_separated_number(row["CTR - Cardiothoracic Ratio"])
        row["Inscribed circle radius"] = convert_comma_separated_number(row["Inscribed circle radius"])
        row["Heart perimeter"] = convert_comma_separated_number(row["Heart perimeter"])

        row["ID"] = int(row["ID"])
        row["Cardiomegaly"] = int(row["Cardiomegaly"])
        row["Heart width"] = int(row["Heart width"])
        row["Lung width"] = int(row["Lung width"])
        row["Heart area"] = int(row["Heart area"])
        row["Lung area"] = int(row["Lung area"])

        row["xx"] = float(row["xx"])
        row["yy"] = float(row["yy"])
        row["xy"] = float(row["xy"])
        row["normalized_diff"] = float(row["normalized_diff"])
        row["Polygon Area Ratio"] = float(row["Polygon Area Ratio"])

        label = row.pop(TARGET_COLUMN)  # Store TARGET_COLUMN value in label variable
        row.pop("ID")  # ID column does not store useful data

        y_data.append(label)  # Append label to the list of labels
        x_data_dicts.append(row)  # Append row to the list of samples

vectorizer = DictVectorizer(sparse=False)  # Converts the list of dicts into a matrix

x = vectorizer.fit_transform(x_data_dicts)  # Ensures a fixed column order required by the model
y = np.array(y_data)  # y assigned to NumPy array to enable indexing in SKF loop

# SKF used to get a reliable accuracy score by averaging N_SPLITS tests (K-Fold Cross-Validation)
SKF = StratifiedKFold(n_splits=N_SPLITS,
                      shuffle=True,
                      random_state=RANDOM_STATE)  # Shuffle data randomly before folding

models_result = {}

# Five different ML models created to test and compare their effectiveness
models = {
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest Classifier": RandomForestClassifier(random_state=RANDOM_STATE),
    "KNeighbors Classifier": KNeighborsClassifier(),  # Without random_state
    "SVC": SVC(random_state=RANDOM_STATE),
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=MAX_ITER),  # max_iter added
}

print(f"Starting Cross-Validation for {len(models)} models...\n")
for model_name, model in models.items():
    fold_scores = {"Accuracy Score": 0,
                   "Precision Score": 0,
                   "Recall Score": 0,
                   "F1 Score": 0}

    # Scale features to similar range of value to use
    # other ML methods such as LogisticRegression,
    # KNeighborsClassifier and SVC
    scaler = StandardScaler()

    for train_index, test_index in SKF.split(x, y):
        # Split data into training and test sets for current fold
        x_train = x[train_index]
        x_test = x[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        # Scale x_train and x_test features
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Train the model and predict results
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)

        # Compare models efficiency using Classification Quality Metrics for the current fold
        # Prevent ZeroDivisionError while calculating results by adding zero_division=0
        acc_scr = accuracy_score(y_test, y_pred)
        prec_scr = precision_score(y_test, y_pred, zero_division=0)
        rec_scr = recall_score(y_test, y_pred, zero_division=0)
        f1_scr = f1_score(y_test, y_pred, zero_division=0)

        fold_scores["Accuracy Score"] += acc_scr
        fold_scores["Precision Score"] += prec_scr
        fold_scores["Recall Score"] += rec_scr
        fold_scores["F1 Score"] += f1_scr

    # Calculate tests mean
    for m, r in fold_scores.items():
        r /= N_SPLITS
        fold_scores[m] = r

    # Assign model score to the fold_scores dictionary
    models_result[model_name] = fold_scores

# Change pandas options to display all table's columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Summary results in table
results = pd.DataFrame.from_dict(models_result, orient='index')
results = results.sort_values(by="Accuracy Score", ascending=False)
print("-----------Average Cross-Validation Scores for Cardiomegaly Prediction-----------")
print(results)

# Save results to csv file
results.to_csv("model_comparison_results.csv")

# KNeighbours Classifier and SVC models performance is the best.
# Code is now correct, but we can improve models performance by adding hyperparamethers.
# GridSearchCV might be used to improve KNeighbours Classifier and SVC models performance.
