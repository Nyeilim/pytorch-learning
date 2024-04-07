from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# 模型构建
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))  # 简单地使用 add 堆叠模型
model.add(Dense(units=1, activation='softmax'))  # Dense 就是 FC 层

# 编译模型，有点像 Java 的正则呢，要编译后才能使用
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# 准备数据，这里是造的模拟数据


x_train = np.random.random((1000, 100))  # 1000 个 100 维的向量作为入参
y_train = np.random.randint(2, size=(1000, 1))  # 1000 个 0 或者 1

# 开始训练
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 训练完后就要进行评估，输入数据(也是造的模拟数据)和标签，输出损失(loss)和精确度(accuracy)
x_test = np.random.random((100, 100))
y_test = np.random.randint(2, size=(100, 1))
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)

