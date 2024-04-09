import tensorflow as tf
import matplotlib.pyplot as plt

# 真实的数据集都是分为训练集和测试集的
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_train[0])

# 模型构建及编译
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)  # 训练
# model.evaluate(x_test, y_test, verbose=2)  # 评估
y = model.predict(x_test)  # 预测
print(y[0])
plt.imshow(x_test[0], cmap=plt.get_cmap('gray_r'))  # 画图
plt.show()
