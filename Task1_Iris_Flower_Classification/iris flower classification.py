import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
iris_data = pd.read_csv("Iris.csv")

print("Dataset Preview:")
print(iris_data.head())

print("\nDataset Details:")
print(iris_data.info())

# Drop ID column if present
if "Id" in iris_data.columns:
    iris_data.drop(columns=["Id"], inplace=True)

# Separate input and output
features = iris_data.drop("Species", axis=1)
target = iris_data["Species"]

# Encode target labels
encoder = LabelEncoder()
encoded_target = encoder.fit_transform(target)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    features,
    encoded_target,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

# Create and train the model
classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(X_train, y_train)

# Predict
predictions = classifier.predict(X_test)

# Evaluate model
print("\nModel Accuracy: {:.2f}%".format(
    accuracy_score(y_test, predictions) * 100
))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(
    iris_data["SepalLengthCm"],
    iris_data["SepalWidthCm"],
    c=encoded_target,
    s=50
)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Dataset Visualization")
plt.grid(True)

plt.show()