import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
#from keras.models import Sequential
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation, Reshape, Lambda, Input, Concatenate
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, ZeroPadding2D
import os
from tensorflow.keras.layers import ELU
from PIL import Image
#from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import ModelCheckpoint


inputShape = (int(1024/2),int(780/2),3)

#args = dict(rotation_range=0, width_shift_range = 0,
#            height_shift_range=0, rescale = 1/255)
#trainDataGen = ImageDataGenerator(**args)
#valDataGen =  ImageDataGenerator(rescale = 1/255)
"""
train = open("train.csv",'r').read().strip().split('\n')
train = [l.split(",") for l in train if l.strip()!='']
val = open("val.csv", 'r').read().strip().split('\n')
val = [l.split(",") for l in val if l.strip()!='']
trainDataGen = CustomDataGen(train, "train", inputShape, 32, 1/255)
valDataGen = CustomDataGen(val, "val", inputShape, 32, 1/255)

train_gen= trainDataGen.flow_from_directory(
    directory = "./train",
    batch_size = 32,
    target_size = (inputShape[0],inputShape[1]),
    shuffle = True
)

val_gen = valDataGen.flow_from_directory(
    directory = "./val",
    batch_size = 32,
    target_size = (inputShape[0],inputShape[1])
)

"""
#from myTransformerModel import Transformer as Encoder
#from myTransformerModel import embeddings as emb



    #make model, return after the two-three convs (see notebook, we want to reduce size somewhat but not completely)
def skip_connect(inputShape, numStart, mid, drop_rate):
    X_in = Input(shape = inputShape)

    X = X_in

    mid = Dense(mid)(X)
    mid = ELU(1.0)(mid)
    mid = Dropout(drop_rate)(mid)

    final = Dense(numStart)(mid)

    final = Dropout(drop_rate)(final) + X

    final = ELU(1.0)(final)

    model = Model(inputs = X_in, outputs = final)

    return model

def model(inputShape, numOut):

    numLayers = 2#continue transformer params here (num_heads, FF size, dropout, etc.)
    kqv_input_dim = 256#changing this b/c don't have linear projection layer
    num_heads = 4
    ff_dim = 512
    drop_rate = 0.05#0.225

    X_input = Input(shape=inputShape, name = "image_in")



    print("Xin shape is " + str(X_input.shape))
    #X = X_input + tf.expand_dims(tf.expand_dims(upcast, 1), 1)
    X = X_input
    print("Now it is " + str(X.shape))

    X = Conv2D(32, kernel_size=7, strides=3, padding="same")(X)
    X = BatchNormalization(axis=3,name="0")(X)
    X = Activation("relu")(X)

    #X = MaxPooling2D(pool_size = (3,3))(X)

    X = Conv2D(64, kernel_size = 5, strides=(4,3), padding="valid")(X)
    X = BatchNormalization(axis=3, name="first")(X)
    X = Activation("relu")(X)#None, k, k, 64

    hold = Conv2D(4, kernel_size=5,strides=5,padding='valid')(X)
    hold = Flatten()(hold)
    hold_ = Dense(hold.shape[1], activation = 'relu')(hold)
    hold = Dense(hold.shape[1])(hold_) + hold
    hold = tf.keras.activations.elu(hold, alpha = 2)

    X = Conv2D(128, kernel_size = 3, strides=3, padding="valid")(X)
    X = BatchNormalization(axis=3, name="second")(X)
    X = Activation("relu")(X)



    X = Conv2D(160, kernel_size = 3, strides=2, padding="valid")(X)
    X = BatchNormalization(axis=3, name="third")(X)
    X = Activation("relu")(X)

    hold_2 = Conv2D(8, kernel_size = 4, strides = 4, padding = 'valid')(X)
    hold_2 = Flatten()(hold_2)
    hold_ = Dense(hold_2.shape[1], activation = 'relu')(hold_2)
    hold_2 = Dense(hold_2.shape[1])(hold_) + hold_2
    hold_2= tf.keras.activations.elu(hold_2, alpha = 2)


    X = Conv2D(200, kernel_size = 3, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3, name="fourth")(X)
    X = Activation("relu")(X)


    X = Conv2D(256, kernel_size = 2, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3, name="fifth")(X)
    X = Activation("relu")(X)

    #X = MaxPooling2D(pool_size = (2,2))(X)
    """
    X = Conv2D(180, kernel_size = 5, strides=4, padding="valid")(X)
    X = BatchNormalization(axis=3, name="second")(X)
    X = Activation("relu")(X)
    """
    #X = Conv2D(4, kernel_size = 1, strides = 1, padding = 'valid')(X)

    #X_patch = tf.image.extract_patches(images = X, sizes = [1, 8, 8, 1], rates = [1,1,1,1], strides = [1,6,6,1], padding = 'VALID', name = 'image_extraction')
    #X_patch_flattened = tf.reshape(X_patch,(-1,X_patch.shape[1]*X_patch.shape[2], X_patch.shape[3]))

    #inputShape2 = X_patch_flattened.shape[1:]

    #print("The shape is " +str(X_patch_flattened.shape))
    #transformer_block = transformerEncoder(inputShape2, numLayers, kqv_input_dim, num_heads, ff_dim, drop_rate)

    #X = transformer_block(X_patch_flattened)


    """

    X = Conv2D(256, kernel_size = 3, strides=2, padding="valid")(X)
    X = BatchNormalization(axis=3, name="third")(X)
    X = Activation("relu")(X)

    X = Conv2D(360, kernel_size = 3, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3, name="fourth")(X)
    X = Activation("relu")(X)

    X = Conv2D(512, kernel_size = 3, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3, name="fifth")(X)
    X = Activation("relu")(X)
    """

    flat = Flatten()(X)


    flat = Dense(512-96, activation = 'relu')(flat)
    flat = Dropout(0.2)(flat)
    flat = Concatenate()([flat, hold, hold_2])

    flat = Dense(512, activation = 'relu')(flat)


    skip1 = skip_connect(flat.shape[1:], 512, 256, 0.25)
    #skip2 = skip_connect(flat.shape[1:], 512, 256, 0.3)


    out = skip1(flat)

    #out = skip2(out)
    out = Dense(256, activation = 'relu')(out)
    out = Dropout(drop_rate+.1)(out)


    out = skip_connect(out.shape[1:], 256, 256, 0.25)(out)

    #out = Concatenate()([out,])

    out = Dropout(drop_rate+.05)(out)



    output = tf.keras.layers.Dense(numOut, activation="softmax")(out)

    model = Model(inputs=[X_input], outputs = output)

    return model

theModel = model(inputShape,8)
"""
theModel.compile(optimizer = "Adam", loss = "categorical_crossentropy",metrics = ['accuracy'])
try:
    visionTrans = True
    if (visionTrans):
        theModel.load_weights("weights/model_weights_vision_transformer")
    else:
        theModel.load_weights("weights/model_weights")
except:
    print("no weights able to be saved")

checkpoint = ModelCheckpoint(filepath="weights/model_weights_vision_transformer",
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode="max")

theModel.fit(train_gen,validation_data = val_gen, verbose = 2, epochs=8, callbacks = [checkpoint])
"""

##pics are about 1024 by 780

