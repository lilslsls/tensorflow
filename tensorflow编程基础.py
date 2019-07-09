        #实例1#
#import tensorflow as tf
#hello=tf.constant('hello ,tensorflow!')

#sess=tf.Session()
#print(sess.run(hello))
#sess.close()

       #实例2#
#import tensorflow as tf
#a=tf.constant(3)
#b=tf.constant(4)
#with tf.Session() as sess:
#    print("相加：%i"%sess.run(a+b))
#    print("相乘：%i"%sess.run(a*b))

       #实例3
import tensorflow as tf
a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)
add=tf.add(a,b)
mul=tf.multiply(a,b)
with tf.Session() as sess:
    print("相加：%i"%sess.run(add,feed_dict={a:3,b:4}))
    print("相乘：%i"%sess.run(mul,feed_dict={a:4,b:6}))
    print(sess.run([mul,add],feed_dict={a:3,b:4}))

       #实例4
