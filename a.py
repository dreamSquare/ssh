import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 1. 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. 数据预处理
# 归一化像素值到0-1范围，并添加通道维度（CNN需要）
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # 形状变为 (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# 将标签转换为one-hot编码
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 3. 构建神经网络模型
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),  # 输入层
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),  # 卷积层
        layers.MaxPooling2D(pool_size=(2, 2)),  # 池化层
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),  # 展平层
        layers.Dropout(0.5),  # 随机失活防止过拟合
        layers.Dense(10, activation="softmax"),  # 输出层
    ]
)

# 4. 编译模型
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# 5. 训练模型
batch_size = 128
epochs = 15
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1  # 使用10%的训练数据作为验证集
)

# 6. 评估模型
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# 7. 保存模型（可选）
model.save("mnist_cnn.keras")
