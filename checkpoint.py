import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plotdata = { "batchsize":[], "loss":[] }

def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

#生成模拟数据
train_X=np.linspace(-1,1,100)
print(train_X.shape)
train_Y=2*train_X+np.random.randn(*train_X.shape)*0.5
plt.plot(train_X,train_Y,'ro',label='Original data')
plt.legend()
plt.show()
tf.reset_default_graph()

# 占位符
X=tf.placeholder("float")
Y=tf.placeholder("float")

#模型参数
W=tf.Variable(tf.random_normal([1]),name="weight")
b=tf.Variable(tf.zeros([1]),name='bias')
#前向结构
Z=tf.multiply(X,W)+b

#反向优化
cost=tf.reduce_mean(tf.square(Y-Z))
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化所有变量
init=tf.global_variables_initializer()
#定义参数
train_epochs=2000
display_step=1

saver=tf.train.Saver(max_to_keep=1)
savedir="log/"

# 启动session
# with tf.Session() as sess:
#     sess.run(init)

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
#                     saver.save(sess,savedir+"linermodel.cpkt",global_step=epoch)

#     print("finished!")
#     print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))

#重启一个图 载入检查点
load_epoch=66   
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())     
    saver.restore(sess2, savedir+"linermodel.cpkt-" + str(load_epoch))
    print ("x=0.2，z=", sess2.run(Z, feed_dict={X: 0.2}))

  
with tf.Session() as sess3:
    sess3.run(tf.global_variables_initializer())     
    ckpt=tf.train.get_checkpoint_state(savedir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess3,ckpt.model_checkpoint_path)
    print ("x=0.2，z=", sess3.run(Z, feed_dict={X: 0.2}))

with tf.Session() as sess4:
    sess4.run(tf.global_variables_initializer())
    kpt=tf.train.latest_checkpoint(savedir)
    if kpt!=None:
        saver.restore(sess4,kpt)
    print ("x=0.2，z=", sess4.run(Z, feed_dict={X: 0.2}))