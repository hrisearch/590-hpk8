
#MODIFIED FROM CHOLLETT P120 
from keras import layers 
from keras import models
import numpy as np
import warnings
warnings.filterwarnings("ignore")
dataset = 'mnist'
NKEEP=10000
batch_size=int(0.05*NKEEP)
epochs=20




#-------------------------------------
#BUILD MODEL SEQUENTIALLY (LINEAR STACK)
#-------------------------------------
if dataset == 'cifar10':
    k = 32
    m = 3
else:
    k = 28
    m = 1

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(k, k, m)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()
#-------------------------------------
#GET DATA AND REFORMAT
#-------------------------------------
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.utils import to_categorical
if dataset == "mnist":
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

elif dataset == "fashion_mnist":
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

elif dataset == "cifar10":
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.reshape((50000, 32, 32, 3))
    test_images = test_images.reshape((10000, 32, 32, 3))

#NORMALIZE
train_images = train_images.astype('float32') / 255 
test_images = test_images.astype('float32') / 255  

#DEBUGGING
print("batch_size",batch_size)
rand_indices = np.random.permutation(train_images.shape[0])
train_images=train_images[rand_indices[0:NKEEP],:,:]
train_labels=train_labels[rand_indices[0:NKEEP]]
# exit()


#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
tmp=train_labels[0]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(tmp, '-->',train_labels[0])
print("train_labels shape:", train_labels.shape)

#-------------------------------------
#COMPILE AND TRAIN MODEL
#-------------------------------------
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)


#-------------------------------------
#EVALUATE ON TEST DATA
#-------------------------------------
train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
print('train_acc:', train_acc)
print('test_acc:', test_acc)

model.save(dataset + '_model.sav')
print(history.history.keys())
acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

import matplotlib.pyplot as plt

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training accuracy and loss')
plt.legend()

#plt.figure()

plt.show()
