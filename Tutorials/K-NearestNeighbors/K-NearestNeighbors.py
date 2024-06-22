from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import numpy as np

# Load the breast cancer dataset
data = load_breast_cancer()

# Print the feature names (attributes) of the dataset
print(data.feature_names)

# Print the target names (classes) of the dataset
print(data.target_names)
# Split the dataset into training and testing sets
# `test_size=0.2` means 20% of the data will be used for testing, and 80% for training
x_train, x_test, y_train, y_test = train_test_split(
    np.array(data.data), np.array(data.target), test_size=0.2
)

# Initialize the K-Nearest Neighbors classifier with 3 neighbors
clf = KNeighborsClassifier(n_neighbors=3)

# Train the classifier using the training data
clf.fit(x_train, y_train)

# Evaluate the classifier on the test data and print the accuracy
print(clf.score(x_test, y_test))
# Example of a dummy data point
new_data_point = np.array(
    [
        14.0,
        20.0,
        90.0,
        600.0,
        0.1,
        0.08,
        0.05,
        0.05,
        0.18,
        0.06,
        0.25,
        1.0,
        1.5,
        50.0,
        0.005,
        0.02,
        0.03,
        0.01,
        0.02,
        0.004,
        16.0,
        25.0,
        110.0,
        800.0,
        0.15,
        0.15,
        0.1,
        0.08,
        0.25,
        0.07,
    ]
)

# Reshape the new data point to have the shape (1, -1) to match the expected input shape of the predict method
new_data_point = new_data_point.reshape(1, -1)

# Predict the class for the new data point
prediction = clf.predict(new_data_point)
print("Predicted class:", data.target_names[prediction[0]])
