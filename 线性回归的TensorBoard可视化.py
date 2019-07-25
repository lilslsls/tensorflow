import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = { "batchsize":[], "loss":[] }

def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

train_X=np.linspace(-1,1,100)
print(train_X.shape)
train_Y=2*train_X+np.random.randn(*train_X.shape)*0.5
plt.plot(train_X,train_Y,'ro',label='Original data')
plt.legend()
plt.show()

tf.reset_default_graph()

X=tf.placeholder("float")
Y=tf.placeholder("float")

#模型参数
W=tf.Variable(tf.random_normal([1]),name="weight")
b=tf.Variable(tf.zeros([1]),name='bias')
#前向结构
Z=tf.multiply(X,W)+b

tf.summary.histogram('z',Z)

#反向优化
cost=tf.reduce_mean(tf.square(Y-Z))
tf.summary.scalar('loss_function',cost)
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化所有变量
init=tf.global_variables_initializer()
#定义参数
train_epochs=20
display_step=2

saver=tf.train.Saver()
savedir="log/"

# 启动session
with tf.Session() as sess:
    sess.run(init)
    #plotdata={"batchsize":[],"loss":[]}#存放批次值和损失值
    #合并所有summary
    merged_summary_op=tf.summary.merge_all()
    #创建summary_writer,用于写文件
    summary_writer=tf.summary.FileWriter('log/mnist_with_summaries',sess.graph)

    #向模型输入数据
    for epoch in range(train_epochs):
        for(x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
            #显示训练中的详细信息
        summary_str=sess.run(merged_summary_op,feed_dict={X:x,Y:y});
        summary_writer.add_summary(summary_str,epoch);#将summary写入文件
        if epoch % display_step==0:
                loss=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
                print("Epoch:",epoch+1,"cost=",loss,"W=", sess.run(W),"b=",sess.run(b))
                if not(loss=="NA"):
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)

    print("finished!")
    saver.save(sess,savedir+"linermodel.cpkt")
    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))

# 载入模型测试
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2,savedir+"linermodel.cpkt")
    print("x=6,z=",sess2.run(Z,feed_dict={X:6}))