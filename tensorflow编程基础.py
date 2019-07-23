        #实例1#
# import tensorflow as tf
# hello=tf.constant('hello ,tensorflow!')

# sess=tf.Session()
# print(sess.run(hello))
# print(hello)
# sess.close()

       #实例2#
# import tensorflow as tf
# a=tf.constant(3)
# b=tf.constant(4)
# with tf.Session() as sess:
#    print("相加：%i"%sess.run(a+b))
#    print("相乘：%i"%sess.run(a*b))

       #实例3
# import tensorflow as tf
# a=tf.placeholder(tf.int16)
# b=tf.placeholder(tf.int16)
# add=tf.add(a,b)
# mul=tf.multiply(a,b)
# with tf.Session() as sess:
#     print("相加：%i"%sess.run(add,feed_dict={a:3,b:4}))
#     print("相乘：%i"%sess.run(mul,feed_dict={a:4,b:6}))


       #实例4
# import tensorflow as tf
# a=tf.placeholder(tf.int16)
# b=tf.placeholder(tf.int16)
# add=tf.add(a,b)
# mul=tf.multiply(a,b)
# with tf.Session() as sess:
#        with tf.device("/gpu:0"):
#               print("相加：%i"%sess.run(add,feed_dict={a:3,b:4}))
#               print("相乘：%i"%sess.run(mul,feed_dict={a:4,b:6}))
#               print(sess.run([mul,add],feed_dict={a:3,b:4}))

      #保存/载入线性回归模型
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# plotdata = { "batchsize":[], "loss":[] }

# def moving_average(a, w=10):
#     if len(a) < w: 
#         return a[:]    
#     return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

# train_X=np.linspace(-1,1,100)
# print(train_X.shape)
# train_Y=2*train_X+np.random.randn(*train_X.shape)*0.5
# plt.plot(train_X,train_Y,'ro',label='Original data')
# plt.legend()
# plt.show()

# tf.reset_default_graph()

# X=tf.placeholder("float")
# Y=tf.placeholder("float")

# #模型参数
# W=tf.Variable(tf.random_normal([1]),name="weight")
# b=tf.Variable(tf.zeros([1]),name='bias')
# #前向结构
# Z=tf.multiply(X,W)+b

# #反向优化
# cost=tf.reduce_mean(tf.square(Y-Z))
# learning_rate=0.01
# optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# #初始化所有变量
# init=tf.global_variables_initializer()
# #定义参数
# train_epochs=20
# display_step=2

# saver=tf.train.Saver()
# savedir="log/"

#启动session
# with tf.Session() as sess:
#     sess.run(init)
#     #plotdata={"batchsize":[],"loss":[]}#存放批次值和损失值
#     #向模型输入数据
#     for epoch in range(train_epochs):
#         for(x,y) in zip(train_X,train_Y):
#             sess.run(optimizer,feed_dict={X:x,Y:y})
#             #显示训练中的详细信息
#         if epoch % display_step==0:
#                 loss=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
#                 print("Epoch:",epoch+1,"cost=",loss,"W=", sess.run(W),"b=",sess.run(b))
#                 if not(loss=="NA"):
#                     plotdata["batchsize"].append(epoch)
#                     plotdata["loss"].append(loss)

#     print("finished!")
#     saver.save(sess,savedir+"linermodel.cpkt")
#     print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))

#载入模型测试
# with tf.Session() as sess2:
#     sess2.run(tf.global_variables_initializer())
#     saver.restore(sess2,savedir+"linermodel.cpkt")
#     print("x=6,z=",sess2.run(Z,feed_dict={X:6}))



#图形显示
    # plt.plot(train_X,train_Y,'ro',label='Original data')
    # plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
    # plt.legend()
    # plt.show()

    # plotdata["avgloss"] = moving_average(plotdata["loss"])
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    # plt.xlabel('Minibatch number')
    # plt.ylabel('Loss')
    # plt.title('Minibatch run vs. Training loss')
     
    # plt.show()

    # print ("x=0.2，z=", sess.run(Z, feed_dict={X: 0.2}))

#模型内容
# import tensorflow as tf
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# saverdir="log/"
# w=tf.Variable(1.0,name="Weight")
# b=tf.Variable(2.0,name="bias")
# cc=tf.Variable(3.1,name="bias2")
# saver=tf.train.Saver({'weight':b,'bias':w,'bias2':cc})
# with tf.Session() as sess:
#        tf.global_variables_initializer().run()
#        saver.save(sess,saverdir+"linermodel.cpkt")
# print_tensors_in_checkpoint_file(saverdir+"linermodel.cpkt", None, True) 
