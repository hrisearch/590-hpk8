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
