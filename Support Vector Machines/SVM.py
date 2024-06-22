from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load the breast cancer dataset
data = load_breast_cancer()
x = data.data  # Feature data
y = data.target  # Target data (labels: malignant or benign)
# Split the data into training and testing sets
# 80% of the data will be used for training, and 20% will be used for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create an instance of the SVM classifier with a linear kernel and C=3
clf = SVC(kernel="linear", C=3)

# Train the SVM classifier on the training data
clf.fit(x_train, y_train)

# Evaluate the SVM classifier on the testing data and print the accuracy
print(f"SVC: {clf.score(x_test, y_test)}")


clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(x_train, y_train)

# Evaluate the KNN classifier on the testing data and print the accuracy
print(f"KNN: {clf2.score(x_test, y_test)}")
