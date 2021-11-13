import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from keras import models
from keras import layers

#GET DATASET
from keras.datasets import cifar10
(X, Y), (test_images, test_labels) = cifar10.load_data()

X = X[:3000]

#NORMALIZE AND RESHAPE
X=X/np.max(X) 



#Injecting Noise for DAE
X2=X+1*np.random.uniform(0,1,X.shape)


X=X.reshape(3000,32, 32, 3)
X2=X2.reshape(3000,32, 32, 3) 

#MODEL
n_bottleneck=3

# SHALLOW
model = models.Sequential()
#model.add(layers.Dense(n_bottleneck, activation='linear', input_shape=(32 * 32 * 3,)))
#model.add(layers.Dense(32*32*3,  activation='linear'))
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32,32,3), padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.add(layers.Conv2D(12, (3, 3), activation='relu', padding='same')) 
model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))



model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(32,32,3), padding='same'))
model.add(layers.UpSampling2D((2, 2)))

model.add(layers.Conv2D(12, (3, 3), activation='relu', padding='same')) 
model.add(layers.UpSampling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))

model.add(layers.Conv2D(1, (3, 3), activation='relu', padding='same'))


# #DEEPER
# model = models.Sequential()
# NH=200
# model.add(layers.Dense(NH, activation='relu', input_shape=(28 * 28,)))
# model.add(layers.Dense(NH, activation='relu'))
# model.add(layers.Dense(n_bottleneck, activation='relu'))
# model.add(layers.Dense(NH, activation='relu'))
# model.add(layers.Dense(NH, activation='relu'))
# model.add(layers.Dense(28*28,  activation='linear'))



#COMPILE AND FIT
model.compile(optimizer='rmsprop',
                loss='mean_squared_error')
model.summary()
print(X.shape)
history = model.fit(X, X, epochs=4, batch_size=1000,validation_split=0.2)

###
# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
###
#EXTRACT MIDDLE LAYER (REDUCED REPRESENTATION)
# from keras import Model 
# extract = Model(model.inputs, model.layers[-2].output) # Dense(128,...)
# X1 = extract.predict(X)
# print(X1.shape)

#2D PLOT
# plt.scatter(X1[:,0], X1[:,1], c=Y, cmap='tab10')
# plt.show()

#3D PLOT
# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=X1[:,0], 
#     ys=X1[:,1], 
#     zs=X1[:,2], 
#     c=Y, 
#     cmap='tab10'
# )
# plt.show()

#PLOT ORIGINAL AND RECONSTRUCTED 
#X1=model.predict(X) 
threshold = 4*model.evaluate(X[:1000], X[:1000], batch_size=X.shape[0])

print('threshold is ')
print(threshold)

from keras.datasets import cifar100 
(Xf, Yf), (test_imagesf, test_labelsf) = cifar100.load_data()

Xf = Xf[:1100]
Yf = Yf[:1100]

#remove trucks
for i in range(len(Yf)):
    if Yf[i] == [58]:
        np.delete(Yf, i)
        np.delete(Xf, i)

Xf = Xf[:1000]
Yf = Yf[:1000]

#NORMALIZE AND RESHAPE
Xf=Xf/np.max(X) 
Xf=Xf.reshape(1000,32,32,3); 


Xf = Xf[:10]
Xf1 = model.predict(Xf)
meanse = np.mean(np.mean(np.mean(np.power(Xf1 - Xf, 2), axis = 1), axis = 1), axis=1) 
print(meanse.shape)
for i in range(len(Xf1)):
    print(meanse[i])
    if meanse[i] > threshold:
        print('anomaly')

# #RESHAPE
# X=X.reshape(60000,28,28); #print(X[0])
# X1=X1.reshape(60000,28,28); #print(X[0])

# #COMPARE ORIGINAL 
# f, ax = plt.subplots(4,1)
# I1=11; I2=46
# ax[0].imshow(X[I1])
# ax[1].imshow(X1[I1])
# ax[2].imshow(X[I2])
# ax[3].imshow(X1[I2])
# plt.show()

