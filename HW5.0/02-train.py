from tensorflow.keras.preprocessing.text import Tokenizer
from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.models import save_model
from keras.layers.convolutional import MaxPooling1D
import numpy as np
with open('book1.txt') as f:
    lines1 = f.readlines()
train1 = lines1[:2400]
test1 = lines1[2400:4800]
val1 = lines1[4800:5200]

with open('book2.txt') as f:
    lines2 = f.readlines()
train2 = lines2[:2400]
test2 = lines2[2400:4800]
val2 = lines2[4800:5200]

with open('book1.txt') as f:
    lines3 = f.readlines()
train3 = lines3[:2400]
test3 = lines3[2400:4800]
val3 = lines3[4800:5200]

# for i in range(239):
#     k = 2*i
#     train1[k:k+10] = [' '.join(train1[k:k+10])]
#     train2[k:k+10] = [' '.join(train2[k:k+10])]
#     train3[k:k+10] = [' '.join(train3[k:k+10])]
#     test1[k:k+10] = [' '.join(test1[k:k+10])]
#     test2[k:k+10] = [' '.join(test2[k:k+10])]
#     test3[k:k+10] = [' '.join(test3[k:k+10])]

train = []
train.extend(train1)
train.extend(train2)
train.extend(train3)
val = []
val.extend(val1)
val.extend(val2)
val.extend(val3)
test = []
test.extend(test1)
test.extend(test2)
test.extend(test3)


ytrain = []
for i in range(len(train1)):
    ytrain.append(1)
for i in range(len(train2)):
    ytrain.append(2)
for i in range(len(train3)):
    ytrain.append(3)


ytest = []
for i in range(len(test1)):
    ytest.append(1)
for i in range(len(test2)):
    ytest.append(2)
for i in range(len(test3)):
    ytest.append(3)

myTokenizer = Tokenizer(num_words=1000)
myTokenizer.fit_on_texts(train)

train_seq = myTokenizer.texts_to_sequences(train)
val_seq = myTokenizer.texts_to_sequences(val)
test_seq = myTokenizer.texts_to_sequences(test)

#print(test_seq)

X_train = train_seq
X_test = test_seq

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

# truncate and pad input sequences
max_review_length = 50
max_length = 300
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

print(np.array(X_train).shape)
print(np.array(ytrain).shape)
vocab_size = len(myTokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=50))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model2 = Sequential()
model2.add(Embedding(vocab_size, 50, input_length=50))
model2.add(LSTM(50))
model2.add(Dense(10, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

value = 'cnn'
if value == 'cnn':
    history = model.fit(X_train, np.array(ytrain), epochs=6)
    print(model.summary())

elif value == 'rnn':
    history = model2.fit(X_train.reshape([X_train.shape[0], 50, 1]).astype(np.float32), np.array(ytrain), epochs=6)
    print(model2.summary())

import pickle
if value == 'cnn':
	model.save('model')
if value == 'rnn':
	model2.save('model')

# import matplotlib.pyplot as plt

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

import pickle
f = open('xt', 'wb')
pickle.dump(X_test, f)
f.close()
f = open('yt', 'wb')
pickle.dump(ytest, f)
f.close()