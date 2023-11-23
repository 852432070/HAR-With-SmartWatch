import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Normalization, ReLU, Dropout
from tensorflow.keras import Model, regularizers
from collections import Counter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow_model_optimization as tfmot

from keras_flops import get_flops

from sklearn.preprocessing import StandardScaler

# Calculate F1-Score while training
from sklearn.metrics import f1_score, recall_score, precision_score

class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return
    
ck_callback = tf.keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_f1:.4f}.hdf5',
                                                 monitor='val_f1', 
                                                 mode='max', verbose=2,
                                                 save_best_only=True,
                                                 save_weights_only=True)
tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch=0)

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

if not os.path.exists('./light_model'):
    os.makedirs('./light_model')


# %%


# PolynomialDecay = pruning_schedule.PolynomialDecay

data_path = '/data/wang_sc/datasets/PAMAP2_Dataset/Processed0/'

# %%
train_x = np.load(data_path + 'x_train.npy').astype(np.float32)
train_y = np.load(data_path + 'y_train.npy').astype(np.int32)
test_x = np.load(data_path + 'x_test.npy').astype(np.float32)
test_y = np.load(data_path + 'y_test.npy').astype(np.int32)
num_classes = len(Counter(train_y.tolist()))
# train_y = tf.one_hot(
# 			indices=train_y, 
# 			depth=train_y.shape[0], 
# 			on_value=1, 
# 			off_value=0, 
# 			axis=-1)
# test_y = tf.one_hot(
# 			indices=test_y, 
# 			depth=test_y.shape[0], 
# 			on_value=1, 
# 			off_value=0, 
# 			axis=-1)

train_shape = train_x.shape
test_shape = test_x.shape
scaler = StandardScaler()
train_x = scaler.fit_transform(
train_x.astype(np.float32).reshape(-1,1)).reshape(train_shape[0], train_shape[1], train_shape[2], 1)
test_x = scaler.transform(
test_x.astype(np.float32).reshape(-1,1)).reshape(test_shape[0], test_shape[1], test_shape[2], 1)


# train_x = train_x.reshape(train_shape[0], train_shape[1], train_shape[2], 1)

# test_x = test_x.reshape(test_shape[0], test_shape[1], test_shape[2], 1)
print(train_x.shape)


train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1024
EPOCHS = 100
LEARNING_RATE = 5e-4

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)




#标准卷积块
def conv_block(
    inputs,
    filters,
    kernel_size=(3,1),
    strides=(1,1)
):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, kernel_regularizer=regularizers.l2(0.01), strides=strides, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    return tf.keras.layers.ReLU()(x)
##深度可分离卷积块
def depthwise_conv_block(
    inputs,
    pointwise_conv_filters,
    strides=(1,1),
    expansion=4
):
    input_channel = inputs.shape[-1]

    x = tf.keras.layers.Conv2D(input_channel * expansion, kernel_size=(1,1), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(input_channel * expansion,(6, 1), padding='same', strides=strides, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    ###深度卷积到此结束



    
    ###下面是逐点卷积
    x = tf.keras.layers.Conv2D(pointwise_conv_filters, kernel_size=(1,1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Dropout(0.2)(x)
    
    atte = tf.keras.layers.GlobalAveragePooling2D()(x)
    atte = Dense(pointwise_conv_filters, activation='relu')(atte)

    x = tf.keras.layers.Multiply()([x, atte])
    avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(x)
    max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(x)
    concat = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    atte = tf.keras.layers.Conv2D(filters = 1, kernel_size=(6,1), padding='same', use_bias=False)(concat)

    x = tf.keras.layers.Multiply()([x, atte])

    identity = tf.keras.layers.Conv2D(pointwise_conv_filters, kernel_size=(1,1), padding='same', strides=strides, use_bias=False, activation='sigmoid')(inputs)
    identity = tf.keras.layers.BatchNormalization()(identity)
    x = tf.keras.layers.ReLU()(x)
    identity = Dropout(0.2)(identity)
    
    return tf.keras.layers.add([x,identity])
 
#mobile_net
def mobilenet_v1(
    inputs,
    classes
):
    channel_size = 8
    ##特征提取层
    x = conv_block(inputs, channel_size, strides=(2,1))
#     x = depthwise_conv_block(x, 64)
#     x = depthwise_conv_block(x, 64, strides=(2,1))
    # x = depthwise_conv_block(x, channel_size*2, strides=(2,1))
#     x = depthwise_conv_block(x, 128)
#     x = depthwise_conv_block(x, 128, strides=(2,1))
    x = depthwise_conv_block(x, channel_size*4, strides=(2,1))
#     x = depthwise_conv_block(x, 256)
#     x = depthwise_conv_block(x, 256, strides=(2,1))
#     x = depthwise_conv_block(x, 256)
    # x = depthwise_conv_block(x, channel_size*8, strides=(2,1))
    x = depthwise_conv_block(x, channel_size*16)
#     x = depthwise_conv_block(x, 512)
#     x = depthwise_conv_block(x, 512)
#     x = depthwise_conv_block(x, 512)
#     x = depthwise_conv_block(x, 1024)
    # x = depthwise_conv_block(x, channel_size*32)
#     x = depthwise_conv_block(x, 1024, strides=(2,1))
#     x = depthwise_conv_block(x, channel_size)
    
    ##全局池化
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    ##全连接层
    pred = tf.keras.layers.Dense(classes, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(x)
    
    return pred
 

 
##模型实例化
inputs = tf.keras.Input(shape=(171,9,1))
model = tf.keras.Model(inputs=inputs, outputs=mobilenet_v1(inputs, num_classes))

validation_split = 0.1 # 10% of training set will be used for validation set. 

num_images = train_x.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / BATCH_SIZE).astype(np.int32) * 5

# prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
# pruning_params = {
#       'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.25,
#                                                                final_sparsity=0.50,
#                                                                begin_step=0,
#                                                                end_step=end_step)
# }


model.summary()
flops = get_flops(model)
print(f"FLOPS: {flops / 10 ** 6:.03} M")
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=['accuracy']
)

model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS,validation_data=(test_x, test_y),
          callbacks=[Metrics(valid_data=(test_x, test_y)),
                     ck_callback,
                     tb_callback])
# model_for_pruning = prune_low_magnitude(model, **pruning_params)
# model_for_pruning.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )
# model_for_pruning.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=5,validation_data=(test_x, test_y),validation_split=validation_split,callbacks=callbacks)

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# @tf.function
# def train_step(images, labels):
#   with tf.GradientTape() as tape:
#     # training=True is only needed if there are layers with different
#     # behavior during training versus inference (e.g. Dropout).
#     predictions = model(images, training=True)
#     loss = loss_object(labels, predictions)
#   gradients = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#   train_loss(loss)
#   train_accuracy(labels, predictions)

# @tf.function
# def test_step(images, labels):
#   # training=False is only needed if there are layers with different
#   # behavior during training versus inference (e.g. Dropout).
#   predictions = model(images, training=False)
#   t_loss = loss_object(labels, predictions)

#   test_loss(t_loss)
#   test_accuracy(labels, predictions)
# accuracy_result = []
# for epoch in range(EPOCHS):
#   # Reset the metrics at the start of the next epoch
#   train_loss.reset_states()
#   train_accuracy.reset_states()
#   test_loss.reset_states()
#   test_accuracy.reset_states()

#   for images, labels in train_dataset:
#     train_step(images, labels)

#   for test_images, test_labels in test_dataset:
#     test_step(test_images, test_labels)

#   print(
#     f'Epoch {epoch + 1}, '
#     f'Loss: {train_loss.result()}, '
#     f'Accuracy: {train_accuracy.result() * 100}, '
#     f'Test Loss: {test_loss.result()}, '
#     f'Test Accuracy: {test_accuracy.result() * 100}'
#   )
#   accuracy_result.append(test_accuracy.result())




# %%
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("./light_model/har_cnn_9axes.tflite", "wb").write(tflite_model)

# %%
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset():
    for i in range(100):
      data = train_x[i].reshape(1,171,9,1)
      yield [data]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8  # or tf.uint8
# converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()
open("./light_model/har_cnn_9axes_q8.tflite", "wb").write(tflite_model)


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset():
    for i in range(100):
      data = train_x[i].reshape(1,171,9,1)
      yield [data]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
# converter.inference_input_type = tf.int8  # or tf.uint8
# converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()
open("./light_model/har_cnn_9axes_q16x8.tflite", "wb").write(tflite_model)
# %%
# model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
# converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# def representative_dataset():
#     for i in range(500):
#       data = test_x[i].reshape(1,171,9,1)
#       yield [data]
# converter.representative_dataset = representative_dataset
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# # converter.inference_input_type = tf.int8  # or tf.uint8
# # converter.inference_output_type = tf.int8  # or tf.uint8
# tflite_model = converter.convert()
# open("/kaggle/working/har_cnn_9axes_pq.tflite", "wb").write(tflite_model)




# %%
import os
size = os.path.getsize("./light_model/har_cnn_9axes.tflite")
size_q = os.path.getsize("./light_model/har_cnn_9axes_q.tflite")

# %%
import matplotlib.pyplot as plt

x=[1,2]  # 确定柱状图数量,可以认为是x方向刻度
y=[size,size_q]  # y方向刻度
params = {
    'figure.figsize': '5, 5'
}
plt.rcParams.update(params)
color=['red','green']
x_label=['Not quantized','quantized']
plt.xticks(x, x_label)  # 绘制x刻度标签
plt.bar(x, y,color=color,width=0.1)  # 绘制y刻度标签


plt.show()

# %%
interpreter = tf.lite.Interpreter(
  model_path="./light_model/har_cnn_9axes_q8.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# %%
# ind = 18

# print(y_test[ind])

# %%
# wave = wave.reshape(171,9,1)
# import time

# input_data = np.expand_dims(wave,axis=0)
num_wave = 500
predicted = 0
for i in range(num_wave):
    wave = test_x[i].reshape(1,171,9,1)
    interpreter.set_tensor(input_details[0]['index'], wave)

#     start_time = time.time()
    interpreter.invoke()
#     stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output_data, axis=1) == test_y[i]:
        predicted+=1
print('Accuracy:', predicted/num_wave)


#test for 16x8 quantization 
interpreter = tf.lite.Interpreter(
  model_path="./light_model/har_cnn_9axes_q16x8.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


num_wave = 500
predicted = 0
for i in range(num_wave):
    wave = test_x[i].reshape(1,171,9,1)
    interpreter.set_tensor(input_details[0]['index'], wave)

#     start_time = time.time()
    interpreter.invoke()
#     stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output_data, axis=1) == test_y[i]:
        predicted+=1
print('Accuracy:', predicted/num_wave)

# %%
# model_for_export.summary()


# # 创建图形和坐标轴对象
# fig, ax = plt.subplots()

# # 绘制折线图
# ax.plot(accuracy_result)

# # 设置图形标题和坐标轴标签
# ax.set_title('Train Process')
# ax.set_xlabel('epoch')
# ax.set_ylabel('accuracy')

# # 显示图形
# plt.show()


