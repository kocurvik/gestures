import os
import cv2
import keras
import numpy as np

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

CLS_DICT = {'put': 0 , 'mov': 1, 'ofo': 2, 'ono': 3, 'palm': 4, 'pao': 5, 'get': 6}

def get_class_number(folder):
    name = os.path.basename(folder)
    possible = [x for x in CLS_DICT.keys() if x in name]
    if len(possible) > 0:
        return CLS_DICT[possible[0]]
    else:
        return None



def load_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # img = cv2.imread(path)
    # img = cv2.resize(img, (224, 224))
    # x = img.astype(keras.backend.floatx())
    # x[..., 0] -= 103.939
    # x[..., 1] -= 116.779
    # x[..., 2] -= 123.68
    return x[0]

import numpy as np
import keras

class SeqDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_path, list, batch_size=2, seq = 6, shuffle=True):
        'Initialization'
        self.seq = seq
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.folders = []
        self.classes = []
        self.end_indices = []

        subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
        for folder in subfolders:
            if int(folder.split('_')[-1]) not in list:
                continue
            cls = get_class_number(folder)
            if cls is None:
                continue
            n_files = int(len([name for name in os.listdir(folder)]) / 2)
            for i in range(8, n_files):
                self.folders.append(folder)
                self.end_indices.append(i)
                self.classes.append(cls)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.folders) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.folders))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indices):
        image_sequences = []
        label_sequences = []
        for idx in indices:
            r_images = []
            r_labels = []
            for k in range(0,self.seq):
                r_name = 'right{}.jpg'.format(self.end_indices[idx]-self.seq+k+1)
                r_img = load_image(os.path.join(self.folders[idx], r_name))
                r_images.append(r_img)
                r_labels.append(self.classes[idx])
            label_sequences.append(np.stack(keras.utils.np_utils.to_categorical(r_labels, num_classes=len(CLS_DICT))))

            image_sequences.append(np.stack(r_images,axis=0))
        X = np.stack(image_sequences,axis=0)
        Y = np.stack(label_sequences,axis=0)
        return X, Y


def load(dataset_path, list, limit):
    X = []
    Y = []
    subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

    for folder in subfolders:
        if int(folder.split('_')[-1]) not in list:
            continue
        cls = get_class_number(folder)
        if cls is None:
            continue
        n_files = int(len([name for name in os.listdir(folder)])/2)
        r_images = []
        l_images = []
        for i in range(limit-2,n_files-limit+2):
            r_name = 'right{}.jpg'.format(i)
            l_name = 'left{}.jpg'.format(i)
            r_img = load_image(os.path.join(folder, r_name))
            l_img = load_image(os.path.join(folder, l_name))
            r_images.append(r_img)
            l_images.append(l_img)

            l = len(r_images)
            if l >= 4:
                r_images_stacked = np.concatenate(l_images[l-4:l], axis = 2)
                l_images_stacked = np.concatenate(l_images[l-4:l], axis = 2)
                X.append(r_images_stacked)
                Y.append(cls)
                X.append(l_images_stacked)
                Y.append(cls)

    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    Y = keras.utils.np_utils.to_categorical(Y)
    return X,Y