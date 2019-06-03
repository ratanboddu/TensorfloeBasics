import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

# Genrating random linear data
# There will be 50 data points ranging from 0 to 50
x = np.linspace(0, 35, 35)
y = np.linspace(0, 35, 35)

# Adding noise to the random linear data
x += np.random.uniform(-3, 3, 35)
y += np.random.uniform(-3, 3, 35)

n = len(x)  # Number of data points

# Plot of Training Data
# plt.scatter(x, y)
# plt.xlabel('x')
# plt.xlabel('y')
# plt.title("Training Data")
# plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(np.random.randn(), name="W")
b = tf.Variable(np.random.randn(), name="b")

learning_rate = 0.01
training_epochs = 1000

# Hypothesis
y_prediction = tf.add(tf.multiply(X, W), b)

# Mean Squared Error Cost Function
cost = tf.reduce_sum(tf.pow(y_prediction - Y, 2)) / (2 * n)

# Gradient Descent Optimizer
gradient_descent_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Global Variables Initializer
init = tf.global_variables_initializer()

# Starting the Tensorflow Session
with tf.Session() as sess:
    # Initializing the Variables
    sess.run(init)

    # Iterating through all the epochs
    for epoch in range(training_epochs):

        # Feeding each data point into the optimizer using Feed Dictionary
        for (_x, _y) in zip(x, y):
            sess.run(gradient_descent_optimizer, feed_dict={X: _x, Y: _y})

            # Displaying the result after every 100 epochs
        if (epoch + 1) % 100 == 0:
            # Calculating the cost of every epoch
            c = sess.run(cost, feed_dict={X: x, Y: y})
            print("Epoch", (epoch + 1), ": Cost = ", c, "W = ", sess.run(W), "b = ", sess.run(b))

    # Storing necessary values to be used outside the Session
    training_cost = sess.run(cost, feed_dict={X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(b)

    predictions = weight * x + bias
    print("Training Cost = ", training_cost, "Weight = ", weight, "Bias = ", bias, '\n')

# Plotting the Results
plt.plot(x, y, 'ro', label ='Original data')
plt.plot(x, predictions, label ='Fitted line')
plt.title('Result - Linear Regression')
plt.legend()
plt.show()
