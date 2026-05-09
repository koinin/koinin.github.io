---
title: "Deeplearning"
date: "2025-02-17T10:04:23+08:00"
draft: true
---

# deep-learning

```python
# 定义网络结构
deeper = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape = [11]),
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])

deeper.summary()

# 预处理
preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)


preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

# 划分训练验证
df_train = red_wine.sample(frac = 0.7, random_state = 0)
df_valid = red_wine.drop(df_train.index)

X_train = df_train.drop('quatity', axis = 1)
X_valid = df_valid_drop('', axis = 1)
y_train = df_train['quatity']
y_valid = df_valid['quatity']

# model.fit

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    verbose=0, # 静默模式
)

# 定义early_stop 防止over under 
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# dropout and batch_normalization
keras.Sequential([
    # ...
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(16),
    # ...
])

# 方法一
layers.Dense(16, activation='relu'),
layers.BatchNormalization(),

# 方法二
layers.Dense(16),
layers.BatchNormalization(),
layers.Activation('relu'),

```

主要流程：
X，y -> preprocessor -> model.Sequential -> add loss and optimizer -> model.fit -> history plot

# CV
> ps:dense model and dense layers 的不同

## Base and Head
```python
# 基础网络示例 (特征提取器)
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',        # 预训练权重
    include_top=False,         # 不包含全连接分类层
    input_shape=(224, 224, 3)  # 输入形状
)

```

```python
# 头部网络示例 (任务特定层)
head = keras.Sequential([
    layers.Flatten(), # 对输入二维转换成一维
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

```

## Conv(Fliter) and relu(detect)
> 主要是tf.nn.conv2d() 和 tf.nn.relu()

The feature extraction performed by the base consists of three basic operations:

1. Filter an image for a particular feature (convolution)
2. Detect that feature within the filtered image (ReLU)
3. Condense the image to enhance the features (maximum pooling)

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3), # activation is None
    # More layers follow
])
```

```python
# 直观看待conv2d 和 relu
# Sympy is a python library for symbolic mathematics. It has a nice
# pretty printer for matrices, which is all we'll use it for.
import sympy
sympy.init_printing()
from IPython.display import display

image = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 0],
])

kernel = np.array([
    [1, -1],
    [1, -1],
])

display(sympy.Matrix(image))
display(sympy.Matrix(kernel))
# Reformat for Tensorflow
image = tf.cast(image, dtype=tf.float32)
image = tf.reshape(image, [1, *image.shape, 1])
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

# conv2d, filters 
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=1,
    padding='VALID',
)
image_detect = tf.nn.relu(image_filter)

# The first matrix is the image after convolution, and the second is
# the image after ReLU.
display(sympy.Matrix(tf.squeeze(image_filter).numpy()))
display(sympy.Matrix(tf.squeeze(image_detect).numpy()))

```

## condense(maximum pooling)

```python
# MAX pooling
import tensorflow as tf

image_condense = tf.nn.pool(
    input=image_detect, # image in the Detect step above
    window_shape=(2, 2),
    pooling_type='MAX',
    # we'll see what these do in the next lesson!
    strides=(2, 2),
    padding='SAME',
)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.show()
```
## Sliding Windows(strides)

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  activation='relu'),
    layers.MaxPool2D(pool_size=2,
                     strides=1,
                     padding='same')
    # More layers follow
])

show_extraction(
    image, kernel,

    # Window parameters
    
    #卷积每次移动距离
    conv_stride=3,
    #池化窗口大小，不影响特征图大小
    pool_size=2,
    #池化窗口每一移动大小，会影响特征图大小，比如此处就是4*4 的 转换成 2*2 
    pool_stride=2,

    subplot_shape=(1, 4),
    figsize=(14, 6),    
)
```

## data augmentation
本质创建更多的数据，通过修图...

```python
# 常见增强
preprocessing.RandomContrast(factor=0.10),
preprocessing.RandomFlip(mode='horizontal'),
preprocessing.RandomRotation(factor=0.10),

# 训练参数
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],# 不影响实验，只作为参考

)
```
