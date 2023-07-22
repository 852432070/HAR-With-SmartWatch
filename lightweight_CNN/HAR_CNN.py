# %%
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Normalization, ReLU, Dropout
from tensorflow.keras import Model
import numpy as np
from collections import Counter
import os
from keras_flops import get_flops
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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


# %%
train_x = np.load('/data/wang_sc/datasets/PAMAP2_Dataset/Processed0/x_train.npy').astype(np.float32)
train_y = np.load('/data/wang_sc/datasets/PAMAP2_Dataset/Processed0/y_train.npy').astype(np.int32)
test_x = np.load('/data/wang_sc/datasets/PAMAP2_Dataset/Processed0/x_test.npy').astype(np.float32)
test_y = np.load('/data/wang_sc/datasets/PAMAP2_Dataset/Processed0/y_test.npy').astype(np.int32)

train_shape = train_x.shape
train_x = train_x.reshape(train_shape[0], train_shape[1], train_shape[2], 1)
test_shape = test_x.shape
test_x = test_x.reshape(test_shape[0], test_shape[1], test_shape[2], 1)
print(train_x.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1024
EPOCHS = 200
LEARNING_RATE = 5e-4

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

num_classes = len(Counter(train_y.tolist()))

# %%
class MyModel(Model):
  def __init__(self, num_classes):
    super(MyModel, self).__init__()
    self.layer1 = self._make_layers(64, (6,1), (3,1))
    self.layer2 = self._make_layers(128, (6,1), (3,1))
    self.layer3 = self._make_layers(256, (6,1), (3,1))
    self.fc1 = Dense(64, activation='relu')
    self.fc2 = Dense(num_classes, activation='softmax')
  def _make_layers(self, output_channel, kernel_size, stride):
    return tf.keras.Sequential([
      Conv2D(output_channel, kernel_size, strides=stride, activation='relu'),
      Normalization(),
      Dropout(0.1)
    ])

  def call(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = Flatten()(x)
    x = self.fc1(x)
    x = Dropout(0.2)(x)
    out = self.fc2(x)
    return out
# Create an instance of the model
model = MyModel(num_classes=num_classes)
model.build(input_shape=(BATCH_SIZE, 171, 9, 1))
model.summary()
# flops = get_flops(model)
# print(f"FLOPS: {flops / 10 ** 6:.03} M")

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=["accuracy"]
)

model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_x, test_y),
          callbacks=[Metrics(valid_data=(test_x, test_y)),
                     ck_callback,
                     tb_callback])
