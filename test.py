import keras
import os
import cv2
import numpy as np
import json
from keras.models import model_from_json

import loader

def convert_to_inference_model(original_model):
    original_model_json = original_model.to_json()
    inference_model_dict = json.loads(original_model_json)

    layers = inference_model_dict['config']
    for layer in layers:
        if 'stateful' in layer['config']:
            layer['config']['stateful'] = True

        if 'batch_input_shape' in layer['config']:
            layer['config']['batch_input_shape'][0] = 1
            layer['config']['batch_input_shape'][1] = None

    inference_model = model_from_json(json.dumps(inference_model_dict))
    inference_model.set_weights(original_model.get_weights())

    return inference_model

if __name__ == '__main__':
    model_path = 'D:/Skola/PhD/code/gestures/snapshots/gestures_lstm_05-1.21-0.61.hdf5'
    model = keras.models.load_model(model_path)
    model = convert_to_inference_model(model)

    dataset_path = 'D:/Skola/PhD/data/gesture_dataset_2018_09_18/dataset/'

    subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

    orig_images = []
    input_images = []

    for folder in subfolders:
        if int(folder.split('_')[-1]) not in [9,10]:
            continue
        n_files = int(len([name for name in os.listdir(folder)])/2)
        r_images = []
        l_images = []
        for i in range(1,n_files):
            r_name = 'right{}.jpg'.format(i)
            r_img = loader.load_image(os.path.join(folder, r_name))
            orig_images.append(cv2.imread(os.path.join(folder, r_name)))
            input_images.append(r_img)


    # image_names = ['getr1_11/right12.jpg', 'putr1_12/right17.jpg', 'palmr_12/right10.jpg', 'paol_11/left20.jpg']

    font = cv2.FONT_HERSHEY_SIMPLEX
    cls_dict = {v: k for k, v in loader.CLS_DICT.items()}
    for i, image in enumerate(orig_images):
        prediction = model.predict(input_images[i][np.newaxis, np.newaxis, :, :, :], batch_size=None, verbose=0, steps=None)

        best = np.argmax(prediction[0][0])
        conf = prediction[0,0,best]
        cls = cls_dict[best]
        image = cv2.putText(image, '{}:{}'.format(cls, conf), (50,50), font, 1,(0,0,255))
        cv2.imshow("Image", image)
        cv2.waitKey(0)
