import tensorflow as tf

hello_world = tf.constant("Valar Morghulis")

sess = tf.Session()

print(sess.run(hello_world))




