import sys
if (not "C:\python310\lib\site-packages" in sys.path):
    sys.path.append("C:\python310\lib\site-packages")
import numpy as np
global X_part


def categorical(instance, arr):
    index = arr.index(instance.strip())
    total = len(arr)
    toRet = [0] * total
    toRet[index] = 1
    return toRet
def one_hot(arr, num_opts):#can leave as func not class b/c both excel in same order
    arr = np.reshape(arr, arr.shape[0])
    if (num_opts == None):
        num_opts = []
        
        for inst in arr:
            if (not (inst.strip() in num_opts)):
                num_opts.append(inst.strip())
    
    toRet = [categorical(i, num_opts) for i in arr]
    return np.array(toRet), num_opts
#pip show

def generateData(fileName, map_back):
    global X_part
    global y_part
    data = open("Datasets/Training.csv",'r')
    d = data.read().split('\n')
    data.close()

    data = [d_.split(',') for d_ in d]


    data = np.array(data)
    X_part = data[0][:-1]
    X = data[1:, :-1]
    X = X.astype(np.int32)
    y = data[1:, -1]
    y = np.reshape(y, (y.shape[0], 1))            
    print("making one hot")
    y, map_back = one_hot(y,map_back)
    y_part = map_back
    return X, y, map_back

X, y, map_back = generateData("Datasets/Training.csv", None)
X_val, y_val, __ = generateData("Datasets/Testing.csv", map_back)
print("finished w/ one_hot")

outputShape = y.shape[1]
inputShape = X.shape[1]


from tensorflow.keras import layers
from tensorflow import keras
def model(inputShape, outputShape):
    inputs = layers.Input(shape = inputShape)

    roundOne = layers.Dense(64, activation = 'relu')(inputs)
    mid = layers.Dense(128, activation = 'relu')(roundOne)
    final = layers.Dense(64)(mid) + roundOne + layers.Dense(64)(inputs)
    final = layers.Activation('relu')(final)

    out = layers.Dense(outputShape, activation = 'softmax')(final)

    return keras.Model(inputs = inputs, outputs = out)
print("Creating model")
print("inputShape is " + str(inputShape))#132
myModel = model(inputShape, outputShape)
myModel.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(myModel.summary())
myModel.fit(X,y, validation_data = (X_val, y_val), batch_size = 256, epochs =4, verbose = 2)
