import tensorflow as tf
import matplotlib.pyplot as plt

x=tf.Variable(3,name="x")
y=tf.Variable(4,name="y")

f=x*x*y +y+2

#print(f)
sess=tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result=sess.run(f)
print(result)
sess.close()