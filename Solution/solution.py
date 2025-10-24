# Import module to read data from csv file
import csv

# Import modules for preprocessing and splitting data
from sklearn.model_selection import train_test_split
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

# Constant variables
FILENAME = "task_data.csv"
TARGET_COLUMN = "Cardiomegaly"
TEST_SIZE = 0.2
RANDOM_STATE = 123  # To guarantee the same results for each program run
MAX_ITER = 1000
N_SPLITS = 5  # Number of CV tests (nr of folds)


# Some strings contain comma instead of dot
def convert_comma_separated_number(number):
    number = number.replace(",", ".")
    return float(number)


x_data_dicts = []  # List of samples in dictionaries
y_data = []  # List of labels

# Extract, preprocess and load data into separate lists of labels and samples

with open(FILENAME, "r") as csvfile:
    data = csv.DictReader(csvfile)  # Load csv file
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

vectorizer = DictVectorizer(sparse=False)  # Converts the list of dicts into a matrix.

x = vectorizer.fit_transform(x_data_dicts)  # Ensures a fixed column order required by the model.
y = np.array(y_data)

# Scale features to similar range of value to use
# other ML methods such as LogisticRegression,
# KNeighborsClassifier and SVC

SKF = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                      random_state=RANDOM_STATE)  # Shuffle data randomly before folding

models_result = {}

models = {
    "DTC_model": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RFC_model": RandomForestClassifier(random_state=RANDOM_STATE),
    "KNC_model": KNeighborsClassifier(),  # WITHOUT RANDOM_STATE
    "SVC_model": SVC(random_state=RANDOM_STATE),
    "LR_model": LogisticRegression(random_state=RANDOM_STATE, max_iter=MAX_ITER),
}

print(f"Starting Cross-Validation for {len(models)} models...")
for model_name, model in models.items():
    fold_scores = {"Accuracy Score": 0,
                   "Precision Score": 0,
                   "Recall Score": 0,
                   "F1 Score": 0}
    for train_index, test_index in SKF.split(x, y):
        x_train = x[train_index]
        x_test = x[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        scaler = StandardScaler()

        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)

        acc_scr = accuracy_score(y_test, y_pred)
        prec_scr = precision_score(y_test, y_pred, zero_division=0)
        rec_scr = recall_score(y_test, y_pred)
        f1_scr = f1_score(y_test, y_pred, zero_division=0)

        fold_scores["Accuracy Score"] += acc_scr
        fold_scores["Precision Score"] += prec_scr
        fold_scores["Recall Score"] += rec_scr
        fold_scores["F1 Score"] += f1_scr

    # Calculate results mean
    for m, r in fold_scores.items():
        r /= N_SPLITS
        fold_scores[m] = r

    models_result[model_name] = fold_scores

print(models_result)

# for model in [DTC_model, RFC_model, KNC_model, SVC_model, LR_model]:
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)

#     # Compare models efficiency using Classification Quality Metrics
#     acc_scr = accuracy_score(y_test, y_pred)
#     prec_scr = precision_score(y_test, y_pred)
#     rec_scr = recall_score(y_test, y_pred)
#     f1_scr = f1_score(y_test, y_pred)

#     print(f"""
#         --------------------------------------
#         ML Method: {model}
#         Accuracy score: {acc_scr}
#         Precision score: {prec_scr}
#         Recall score: {rec_scr}
#         F1 Score: {f1_scr}
#         --------------------------------------
#     """)
