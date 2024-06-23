import cv2 as cv  # to import our own images
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf  # in order to build the neural network to get the data and test it

# Load the MNIST dataset: data set with all handwritten digits
mnist = tf.keras.datasets.mnist

# Split the data 10%-20% for testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data (scale pixel values to range [0, 1])
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# Define the model
model = tf.keras.models.Sequential()

# Input Layer
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# Dense Layer
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# Output Layer
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
# Train the model
model.fit(
    x_train, y_train, epochs=4
)  # epochs: how many time is the model going to see the data again (repeat the process)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")

# Save the model with the .keras extension
model.save("digits.keras")
# Open the Images
for i in range(1, 11):
    img = cv.imread(f"numbers/{i}.jpg")[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"The result is probably {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
