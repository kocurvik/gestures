import keras
import keras_resnet
import keras_resnet.models
import loader
import numpy as np
from keras.layers import TimeDistributed, Dense, LSTM
from keras.models import Sequential

def create_callbacks():
    callbacks = []
    checkpoint = keras.callbacks.ModelCheckpoint('snapshots/gestures_lstm_{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5', verbose=1)
    callbacks.append(checkpoint)

    tb_callback = keras.callbacks.TensorBoard(log_dir='./logs/run_lstm_1')
    callbacks.append(tb_callback)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, min_lr = 0.00001)
    callbacks.append(reduce_lr)

    return callbacks

if __name__ == '__main__':
    classes = len(loader.CLS_DICT)
    seq = 6
    batch = 2

    train_generator = loader.SeqDataGenerator('D:\Skola\PhD\data\gesture_dataset_2018_09_18\dataset',[1,2,3,4,5,6,7,8],
                                              batch_size= batch, seq=seq)
    val_generator = loader.SeqDataGenerator('D:\Skola\PhD\data\gesture_dataset_2018_09_18\dataset',[9,10],
                                            batch_size=batch, seq=seq)

    # get the model
    input_layer = keras.layers.Input((seq,224,224,3))

    # model = keras_resnet.models.ResNet18(input_layer, classes=classes, freeze_bn = False)

    model = Sequential()

    resnet_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet',
                                         pooling='avg')

    model.add(TimeDistributed(resnet_model, input_shape=(seq,224,224,3)))

    model.add(TimeDistributed(Dense(128, activation='relu')))
    # model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(classes, return_sequences=True, activation='softmax'))

    # model = keras.Model(inputs=input_layer, outputs=predictions)
    adam = keras.optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    # prepare training data
    # train_X,train_y = loader.load('D:\Skola\PhD\data\gesture_dataset_2018_09_18\dataset',[1,2,3,4,5,6,7,8], 5)
    # train_generator = loader.generator([1,2,3,4,5,6,7,8])


    # prepare val_data
    # val_X,val_y = loader.load('D:\Skola\PhD\data\gesture_dataset_2018_09_18\dataset',[9,10], 5)

    callbacks = create_callbacks()
    model.fit_generator(train_generator, epochs=60, validation_data=val_generator, callbacks=callbacks)

