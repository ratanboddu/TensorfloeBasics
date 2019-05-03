import tensorflow as tf

a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

# Function for adding, multiplying,  division, subtraction, modulus
add = tf.add(a, b)
multiply = tf.multiply(a, b)
division = tf.divide(a, b)
subtract = tf.subtract(a, b)
modulus = tf.mod(a, b)

with tf.Session() as sess:
    print("Addition = %i" % sess.run(add, feed_dict={a: 5, b: 5}))
    print("Multiplication = %i" % sess.run(multiply, feed_dict={a: 5, b: 5}))
    print("Division = %i" % sess.run(division, feed_dict={a: 5, b: 5}))
    print("Subtraction = %i" % sess.run(subtract, feed_dict={a: 5, b: 5}))
    print("Modulus = %i" % sess.run(modulus, feed_dict={a: 5, b: 5}))

