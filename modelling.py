
import os
import numpy as np
import tensorflow as tf
from preprocessing import preprocess_raw_csi_data,load_csi_data_from_numpy_data
from model_train import build_att_bilstm_cnn_model,train_valid_split,load_model

data_files = os.listdir("/Users/vamshireddy/Downloads/SPRING2022/CSC275/CSC275-LSTM-CNN-HAR/input_files")
data_files = [os.path.join("/Users/vamshireddy/Downloads/SPRING2022/CSC275/CSC275-LSTM-CNN-HAR/input_files",file) for file in data_files]

numpy_tuple = load_csi_data_from_numpy_data(data_files)

#numpy_tuple = preprocess_raw_csi_data('Dataset/Data/', save=False)

x_bed, y_bed, x_fall, y_fall, x_pickup, y_pickup, x_run, y_run, x_sitdown, y_sitdown, x_standup, y_standup, x_walk, y_walk = numpy_tuple
x_train, y_train, x_valid, y_valid = train_valid_split(
    (x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk),
    train_portion=0.9, seed=70)

# parameters for Deep Learning Model
# lstm_units = 200
# attention_units = 400
model = build_att_bilstm_cnn_model()

# train
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.summary()

model.fit(
    x_train,
    y_train,
    batch_size=128, epochs=20,
    validation_data=(x_valid, y_valid),
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('best_atten_withCNN_temp.hdf5',
                                            monitor='val_accuracy',
                                            save_best_only=True,
                                            save_weights_only=False)
        ])

#evaluation
# load the best model
# model = load_model('/Users/vamshireddy/Downloads/SPRING2022/CSC275/CSC275-LSTM-CNN-HAR/atten_BILSTM_CNN.hdf5')
# y_pred = model.predict(x_valid)
#
# from sklearn.metrics import confusion_matrix,accuracy_score
#
# print(accuracy_score(np.argmax(y_valid, axis=1), np.argmax(y_pred, axis=1)))
# print(confusion_matrix(np.argmax(y_valid, axis=1), np.argmax(y_pred, axis=1)))