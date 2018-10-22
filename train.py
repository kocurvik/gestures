import keras
import keras_resnet
import keras_resnet.models
import loader
import numpy as np

def create_callbacks():
    callbacks = []
    checkpoint = keras.callbacks.ModelCheckpoint('snapshots/gestures_multi_{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5', verbose=1)
    callbacks.append(checkpoint)

    tb_callback = keras.callbacks.TensorBoard(log_dir='./logs/run_2')
    callbacks.append(tb_callback)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, min_lr = 0.00001)
    callbacks.append(reduce_lr)

    return callbacks

if __name__ == '__main__':
    classes = len(loader.CLS_DICT)

    # get the model
    pseudo_input_layer = keras.layers.Input((224,224,12))
    input_layer = keras.layers.Conv2D(3,(1,1))(pseudo_input_layer)

    model = keras.models.load_model('D:/Skola/PhD/code/gestures/snapshots/gestures_01-1.49-0.71.hdf5')(input_layer)


    # model = keras_resnet.models.ResNet18(input_layer, classes=classes, freeze_bn = False)

    # resnet_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=input_layer,
    #                                      pooling='avg')

    # for layer in resnet_model.layers:
    #     layer.trainable = False

    # x = resnet_model.output
    # # x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Dense(1024, activation='relu')(x)
    # predictions = keras.layers.Dense(classes, activation='softmax')(x)

    model = keras.Model(inputs=pseudo_input_layer, outputs=model)
    # model = keras.Model(inputs=input_layer, outputs=predictions)
    adam = keras.optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # prepare training data
    train_X,train_y = loader.load('D:\Skola\PhD\data\gesture_dataset_2018_09_18\dataset',[1,2,3,4,5,6,7,8], 5)

    # prepare val_data
    val_X,val_y = loader.load('D:\Skola\PhD\data\gesture_dataset_2018_09_18\dataset',[9,10], 5)

    print(val_X.shape)
    print(train_X.shape)

    callbacks = create_callbacks()

    model.fit(x = train_X, y = train_y, batch_size=16, epochs=60, validation_data=(val_X, val_y), callbacks=callbacks)

