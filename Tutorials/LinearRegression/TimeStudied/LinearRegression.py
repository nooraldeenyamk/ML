import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data
time_studied = np.array([18, 60, 34, 53, 23, 12, 54, 67, 23, 44]).reshape(-1, 1)
scores = np.array([46, 98, 68, 87, 56, 40, 85, 100, 56, 95]).reshape(-1, 1)

# Print the input data
# print(time_studied)
# Create and train the model
model = LinearRegression()
model.fit(time_studied, scores)

# Prepare the data for plotting the regression line
x_new = np.linspace(0, 70, 100).reshape(-1, 1)

# Plot the data and the regression line
plt.scatter(time_studied, scores)
plt.plot(
    x_new,
    model.predict(x_new),
    "r",
)
plt.ylim(0, 100)
plt.xlabel("Time Studied")
plt.ylabel("Scores")
plt.title("Time Studied vs. Scores with Linear Regression Line")
plt.show()

# Predict the score for x hours of study time
print(model.predict(np.array([5]).reshape(-1, 1)))
time_train, time_test, score_train, score_test = train_test_split(
    time_studied, scores, test_size=0.3
)
model = LinearRegression()

model.fit(time_train, score_train)
model.score(time_test, score_test)

print(model.score(time_test, score_test))

plt.scatter(time_train, score_train)
plt.plot(
    np.linspace(0, 70, 100).reshape(-1, 1),
    model.predict(np.linspace(0, 70, 100).reshape(-1, 1)),
    "r",
)
plt.show()
