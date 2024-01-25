# %%
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Normalization, ReLU, Dropout
from tensorflow.keras import Model
import numpy as np
from collections import Counter
import os
from keras_flops import get_flops
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

data_path = '/data/wang_sc/datasets/PAMAP2_Dataset/Processed_self_made/'

# %%
train_x = np.load(data_path + 'x_train.npy').astype(np.float32)
train_y = np.load(data_path + 'y_train.npy').astype(np.int32)
test_x = np.load(data_path + 'x_test.npy').astype(np.float32)
test_y = np.load(data_path + 'y_test.npy').astype(np.int32)

train_shape = train_x.shape
test_shape = test_x.shape
scaler = StandardScaler()
train_x = scaler.fit_transform(
train_x.astype(np.float32).reshape(-1,1)).reshape(train_shape[0], train_shape[1], train_shape[2], 1)
test_x = scaler.transform(
test_x.astype(np.float32).reshape(-1,1)).reshape(test_shape[0], test_shape[1], test_shape[2], 1)
print(train_x.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1024
EPOCHS = 100
LEARNING_RATE = 5e-4

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

num_classes = len(Counter(train_y.tolist()))

def make_layers(inputs, output_channel, kernel_size, stride):
    input_channel = inputs.shape[-1]

    

    x = tf.keras.layers.Conv2D(input_channel,(6, 1), padding='same', strides=stride, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(output_channel, kernel_size=(1,1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # identity = tf.keras.layers.Conv2D(output_channel, kernel_size=(1,1), padding='same', strides=stride, use_bias=False)(inputs)
    # identity = tf.keras.layers.BatchNormalization()(identity)
    # identity = Dropout(0.2)(identity)
    
    # return tf.keras.layers.add([x,identity])
    return x


def resnet(
    inputs,
    classes
):
  channel_size = 8
  x = make_layers(inputs, channel_size, (6,1), (2,1))
  x = make_layers(x, channel_size * 2 * 4, (6,1), (2,1))
  x = make_layers(x, channel_size * 4 * 4, (6,1), (2,1))
  x = make_layers(x, channel_size * 8 * 4, (6,1), (2,1))
  x = make_layers(x, channel_size * 16 * 4, (6,1), (2,1))
  x = make_layers(x, channel_size * 32 * 4, (6,1), (2,1))

  x = Flatten()(x)
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.2)(x)
  out = Dense(classes, activation='softmax')(x)
  return out

# Create an instance of the model
inputs = tf.keras.Input(shape=(171,9,1))
model = tf.keras.Model(inputs=inputs, outputs=resnet(inputs, num_classes))
model.summary()
flops = get_flops(model)
print(f"FLOPS: {flops / 10 ** 6:.03} M")

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["accuracy"]
)

model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_x, test_y),
          callbacks=[Metrics(valid_data=(test_x, test_y)),
                     ck_callback,
                     tb_callback])


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset():
    for i in range(100):
      data = train_x[i].reshape(-1,171,9,1)
      yield [data]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
# converter.inference_input_type = tf.int8  # or tf.uint8
# converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()
open("./light_model/har_dwcnn_9axes_q.tflite", "wb").write(tflite_model)

interpreter = tf.lite.Interpreter(
  model_path="./light_model/har_dwcnn_9axes_q.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


num_wave = 4000
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