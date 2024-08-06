import sys

import os
import tensorflow as tf
import sys 
import json
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import_model_paths = sys.argv[1]
# import_model_paths = 'ResNet152V2'
# import_model_paths = 'ResNet101V2'
# import_model_paths = 'ResNet50V2'
print(import_model_paths)
# quit()
resize_size = (250, 250)
MODEL_FOLDER = os.path.join("models")
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER, exist_ok=True)

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img, channels=3)
    return img

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
    return [label['keypoints']]

def main(args):
    epoch_num = 300 
    for VIEW in ['AP', 'LA']:
        DATASET_NAME = 'BUU_LSPINE_V2_' + VIEW
        IMAGE_PATHS = {
            'train': os.path.join('data', DATASET_NAME, 'train'),
            'test': os.path.join('data', DATASET_NAME, 'test'),
            'eval': os.path.join('data', DATASET_NAME, 'eval'),
        }

        images = {}
        for i_type in IMAGE_PATHS:
            images[i_type] = tf.data.Dataset.list_files(os.path.join(IMAGE_PATHS[i_type], 'images','*.jpg'), shuffle=False)
            images[i_type] = images[i_type].map(load_image)
            images[i_type] = images[i_type].map(lambda x: tf.image.resize(x, resize_size))
            images[i_type] = images[i_type].map(lambda x: x/255)

        labels = {}
        for i_type in IMAGE_PATHS:
            labels[i_type] = tf.data.Dataset.list_files(os.path.join(IMAGE_PATHS[i_type], 'labels', '*.json'), shuffle=False)
            labels[i_type] = labels[i_type].map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

        data = {}
        for i_type in ['train', 'test', 'eval']:
            data[i_type] = tf.data.Dataset.zip((images[i_type], labels[i_type]))
            data[i_type] = data[i_type].shuffle(len(os.path.join(IMAGE_PATHS[i_type], 'images','*.jpg')))
            data[i_type] = data[i_type].batch(16)
            data[i_type] = data[i_type].prefetch(4)

        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import Input, Conv2D, Reshape, Dropout
        output_size_map ={
            'AP':40,
            'LA':44
        }
        if import_model_paths == 'ResNet152V2':
            from tensorflow.keras.applications import ResNet152V2
            # Create Model
            MODEL_NAME = import_model_paths
            MODEL_PATH = os.path.join(MODEL_FOLDER, VIEW + '_' + MODEL_NAME + '.h5')
            model = Sequential([
                Input(shape=(250,250,3)), 
                ResNet152V2(include_top=False, input_shape=(250,250,3)),
                Conv2D(512, 3, padding='same', activation='relu'),
                Conv2D(512, 3, padding='same', activation='relu'),
                Conv2D(256, 3, 2, padding='same', activation='relu'),
                Conv2D(256, 2, 2, activation='relu'),
                Dropout(0.05),
                Conv2D(output_size_map[VIEW], 2, 2),
                Reshape((output_size_map[VIEW],))
            ])
        if import_model_paths == 'ResNet101V2':
            from tensorflow.keras.applications import ResNet101V2
            # Create Model
            MODEL_NAME = import_model_paths
            MODEL_PATH = os.path.join(MODEL_FOLDER, VIEW + '_' + MODEL_NAME + '.h5')
            model = Sequential([
                Input(shape=(250,250,3)), 
                ResNet101V2(include_top=False, input_shape=(250,250,3)),
                Conv2D(512, 3, padding='same', activation='relu'),
                Conv2D(512, 3, padding='same', activation='relu'),
                Conv2D(256, 3, 2, padding='same', activation='relu'),
                Conv2D(256, 2, 2, activation='relu'),
                Dropout(0.05),
                Conv2D(output_size_map[VIEW], 2, 2),
                Reshape((output_size_map[VIEW],))
            ])
        if import_model_paths == 'ResNet50V2':
            from tensorflow.keras.applications import ResNet50V2
            # Create Model
            MODEL_NAME = import_model_paths
            MODEL_PATH = os.path.join(MODEL_FOLDER, VIEW + '_' + MODEL_NAME + '.h5')
            model = Sequential([
                Input(shape=(250,250,3)), 
                ResNet50V2(include_top=False, input_shape=(250,250,3)),
                Conv2D(512, 3, padding='same', activation='relu'),
                Conv2D(512, 3, padding='same', activation='relu'),
                Conv2D(256, 3, 2, padding='same', activation='relu'),
                Conv2D(256, 2, 2, activation='relu'),
                Dropout(0.05),
                Conv2D(output_size_map[VIEW], 2, 2),
                Reshape((output_size_map[VIEW],))
            ])
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        loss = tf.keras.losses.MeanAbsoluteError()
        model.compile(optimizer, loss)
        hist = model.fit(data['train'], epochs=epoch_num, validation_data=data['eval'])
        model.save(MODEL_PATH) 
        

if __name__ == "__main__":
    main(sys.argv[1:])  # sys.argv[0] is the script name itself
#     # print(images)
#     # print(labels)
#     # print(data)

quit()
resize_size = (250, 250)
VIEW = 'LA'
# GENDER = 'M'
for GENDER in ['M','F']:
    # for all
    if GENDER == 'M':
        continue
    GENDER = ""
    # --------
    DATASET_NAME = "BUU_LSPINE_V2_" + VIEW
    IMAGE_PATHS = {
        'train': os.path.join('data', DATASET_NAME, 'train'),
        'test': os.path.join('data', DATASET_NAME, 'test'),
        'eval': os.path.join('data', DATASET_NAME, 'eval'),
    }

    images = {}
    for i_type in IMAGE_PATHS:
        # print(glob(os.path.join(IMAGE_PATHS[i_type], 'images','*'+GENDER+'*.jpg')))

        images[i_type] = tf.data.Dataset.list_files(os.path.join(IMAGE_PATHS[i_type], 'images','*'+GENDER+'*.jpg'), shuffle=False)
        images[i_type] = images[i_type].map(load_image)
        images[i_type] = images[i_type].map(lambda x: tf.image.resize(x, resize_size))
        images[i_type] = images[i_type].map(lambda x: x/255)

    labels = {}
    for i_type in IMAGE_PATHS:
        labels[i_type] = tf.data.Dataset.list_files(os.path.join(IMAGE_PATHS[i_type], 'labels', '*'+GENDER+'*.json'), shuffle=False)
        labels[i_type] = labels[i_type].map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

    data = {}
    for i_type in ['train', 'test', 'eval']:
        data[i_type] = tf.data.Dataset.zip((images[i_type], labels[i_type]))
        data[i_type] = data[i_type].shuffle(len(os.path.join(IMAGE_PATHS[i_type], 'images','*'+GENDER+'*.jpg')))
        data[i_type] = data[i_type].batch(16)
        data[i_type] = data[i_type].prefetch(4)


    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Input, Conv2D, Reshape, Dropout
    
    if VIEW == 'AP':
        output_size = 40
    if VIEW == 'LA':
        output_size = 44
    if import_model_paths == 'ResNet152V2':
        from tensorflow.keras.applications import ResNet152V2
        # Create Model
        MODEL_NAME = import_model_paths
        MODEL_PATH = os.path.join('models', GENDER+'_' + VIEW + '_' + MODEL_NAME + '.h5')
        model = Sequential([
            Input(shape=(250,250,3)), 
            ResNet152V2(include_top=False, input_shape=(250,250,3)),
            Conv2D(512, 3, padding='same', activation='relu'),
            Conv2D(512, 3, padding='same', activation='relu'),
            Conv2D(256, 3, 2, padding='same', activation='relu'),
            Conv2D(256, 2, 2, activation='relu'),
            Dropout(0.05),
            Conv2D(output_size, 2, 2),
            # Reshape((40,)),
            Reshape((output_size,))
        ])
        print(model.summary())

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # loss = tf.keras.losses.MeanSquaredError()
        loss = tf.keras.losses.MeanAbsoluteError()
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0007)
        # loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer, loss)
    if import_model_paths == 'ResNet101V2':
        from tensorflow.keras.applications import ResNet101V2
        # Create Model
        MODEL_NAME = import_model_paths
        MODEL_PATH = os.path.join('models', GENDER+'_' + VIEW + '_' + MODEL_NAME + '.h5')
        model = Sequential([
            Input(shape=(250,250,3)), 
            ResNet101V2(include_top=False, input_shape=(250,250,3)),
            Conv2D(512, 3, padding='same', activation='relu'),
            Conv2D(512, 3, padding='same', activation='relu'),
            Conv2D(256, 3, 2, padding='same', activation='relu'),
            Conv2D(256, 2, 2, activation='relu'),
            Dropout(0.05),
            Conv2D(output_size, 2, 2),
            # Reshape((40,)),
            Reshape((output_size,))
        ])
        print(model.summary())

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # loss = tf.keras.losses.MeanSquaredError()
        loss = tf.keras.losses.MeanAbsoluteError()
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0007)
        # loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer, loss)
    if import_model_paths == 'ResNet50V2':
        from tensorflow.keras.applications import ResNet50V2
        # Create Model
        MODEL_NAME = import_model_paths
        MODEL_PATH = os.path.join('models', GENDER+'_' + VIEW + '_' + MODEL_NAME + '.h5')
        model = Sequential([
            Input(shape=(250,250,3)), 
            ResNet50V2(include_top=False, input_shape=(250,250,3)),
            Conv2D(512, 3, padding='same', activation='relu'),
            Conv2D(512, 3, padding='same', activation='relu'),
            Conv2D(256, 3, 2, padding='same', activation='relu'),
            Conv2D(256, 2, 2, activation='relu'),
            Dropout(0.05),
            Conv2D(output_size, 2, 2),
            # Reshape((40,)),
            Reshape((output_size,))
        ])
        print(model.summary())

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # loss = tf.keras.losses.MeanSquaredError()
        loss = tf.keras.losses.MeanAbsoluteError()
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0007)
        # loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer, loss)
    hist = model.fit(data['train'], epochs=300, validation_data=data['eval'])
    model.save(MODEL_PATH) 

    HISTORY_FILE_PATH = os.path.join('history', GENDER+'_' + VIEW + '_' + 'hist_' + MODEL_NAME + '.json')
    # HISTORY_FILE_PATH = os.path.join('history', 'All_' + 'hist_' + MODEL_NAME + '.json')
    try:
        with open(HISTORY_FILE_PATH,"r") as file:
            old_hist = json.load(file)
        hist.history['loss'] = old_hist['loss'] + hist.history['loss']
        hist.history['val_loss'] = old_hist['val_loss'] + hist.history['val_loss']
        # Save the history
        with open(HISTORY_FILE_PATH, 'w') as f:
            json.dump(hist.history, f)
    except:
        with open(HISTORY_FILE_PATH, 'w') as f:
            json.dump(hist.history, f)



