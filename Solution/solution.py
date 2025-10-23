import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

FILENAME = "task_data.csv"
TARGET_COLUMN = "Cardiomegaly"
TEST_SIZE = 0.2
RANDOM_STATE = 123


# Some strings contain comma instead of dot
def convert_comma_separated_number(number):
    number = number.replace(",", ".")
    return float(number)


x_data_dicts = []  # List of samples in dictionaries
y_data = []  # List of labels

# Extract data, convert data type and load into

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


vectorizer = DictVectorizer(sparse=False)

x = vectorizer.fit_transform(x_data_dicts)
y = y_data

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

